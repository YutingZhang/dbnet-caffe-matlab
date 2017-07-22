#include <cstring>
#include <map>
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include <list>

namespace caffe {

StoredSyncedMemory::~StoredSyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

inline void StoredSyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void StoredSyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* StoredSyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void StoredSyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* StoredSyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void StoredSyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* StoredSyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* StoredSyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void StoredSyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

// -------------------------------------

LockedSyncedMemory::LockedSyncedMemory( synced_mem_with_mutex smwm ) 
    : SyncedMemory(INIT_FROM_BASE(),*(smwm.first)), lock_(*(smwm.second),boost::try_to_lock) {
}

SyncedMemoryAllocator::SyncedMemoryAllocator() : 
    m_(new boost::mutex), tag("") { }
SyncedMemoryAllocator::SyncedMemoryAllocator(const std::string& tag_str) : 
    m_(new boost::mutex), tag(tag_str) { }

boost::shared_ptr<LockedSyncedMemory> SyncedMemoryAllocator::alloc( size_t s ) {

	boost::mutex::scoped_lock lock(*(this->m_));

#ifndef CPU_ONLY
    int device_id;
    cudaGetDevice(&device_id);
#else
    const int device_id = -1;
#endif

    mem_bank_t& b = gb_[device_id];


	if (!b.empty()) {
		mem_bank_t::iterator largest_avail = b.end();
		for ( mem_bank_t::iterator iter=b.begin(); iter!=b.end(); ++iter ) {
	        boost::shared_ptr<LockedSyncedMemory> c( new LockedSyncedMemory( *iter ) );
            // this must be released before b.erase; otherwise, the lock will try to release a non-existing mutex
            // so do not define c outside (always keep variable as local as possible to avoid weird errors)
			if (c->is_locked()) {
				if ( iter->first->size() >= s )
					return c;
				largest_avail=iter; // note that the multiset is ordered
			}
        }
		if ( largest_avail!=b.end() ) {
            // do not change content directly. Remove it keep the ordering correct
			b.erase( largest_avail );
		}
	}

	{
		synced_mem_with_mutex t;
		t.first.reset( new SyncedMemory( s ) );
        t.second.reset( new boost::mutex );
		mem_bank_t::iterator cur = b.insert(t);
	    boost::shared_ptr<LockedSyncedMemory> c( new LockedSyncedMemory( *cur ));
		CHECK(c->is_locked()) << "Interal error: the newly created memory cannot be locked";
		return c;
	}

}

SyncedMemoryGuard DynamicSyncedMemory::hold() {
    return SyncedMemoryGuard( new SyncedMemoryGuard_Content_Dynamic( *this ) );
}

const void* DynamicSyncedMemory::cpu_data() {
	CHECK((bool)bsm_) << "request access to non-held dynamic synced memory";
	return bsm_->cpu_data();
}
const void* DynamicSyncedMemory::gpu_data() {
	CHECK((bool)bsm_) << "request access to non-held dynamic synced memory";
	return bsm_->gpu_data();
}
void* DynamicSyncedMemory::mutable_cpu_data() {
	CHECK((bool)bsm_) << "request access to non-held dynamic synced memory";
	return bsm_->mutable_cpu_data();
}
void* DynamicSyncedMemory::mutable_gpu_data() {
	CHECK((bool)bsm_) << "request access to non-held dynamic synced memory";
	return bsm_->mutable_gpu_data();
}

void DynamicSyncedMemory::transfer_guard_to( DynamicSyncedMemory& dsm ) {
	std::list<SyncedMemoryGuard_Content_Dynamic*> gl(guard_set_.begin(),guard_set_.end());
	for ( auto ge : gl ) {
		ge->change_owner(dsm);
	}
}

DynamicSyncedMemory::~DynamicSyncedMemory() {
	std::list<SyncedMemoryGuard_Content_Dynamic*> gl(guard_set_.begin(),guard_set_.end());
	for ( auto ge : gl ) {
		ge->release_owner();
	}
}

SyncedMemoryGuard_Content_Dynamic::SyncedMemoryGuard_Content_Dynamic( DynamicSyncedMemory& dsm ) : dsm_(NULL) {
	boost::mutex::scoped_lock lock1(m_);
    set_owner( dsm );
}

SyncedMemoryGuard_Content_Dynamic::~SyncedMemoryGuard_Content_Dynamic() {
	boost::mutex::scoped_lock lock1(m_);
    release_owner();
}

void SyncedMemoryGuard_Content_Dynamic::change_owner( DynamicSyncedMemory& dsm ) {
	boost::mutex::scoped_lock lock1(m_);
	release_owner();
	set_owner( dsm );
}

void SyncedMemoryGuard_Content_Dynamic::set_owner( DynamicSyncedMemory& dsm ) {
	dsm_ = &dsm;
	boost::mutex::scoped_lock lock(dsm_->m_);
    if (dsm_->guard_set_.empty()) {
        // LOG(INFO) << "Hold DSM: Alloc";
	    dsm_->bsm_holder_ = dsm_->sma_.alloc(dsm_->size_);
	    dsm_->bsm_ = dsm_->bsm_holder_.get();
        // LOG(INFO) << "Hold DSM: Alloc: DONE";
    }
    //LOG(INFO) << "Hold DSM: " << dsm_;
    dsm_->guard_set_.insert(this);
    //LOG(INFO) << "dsm_->bsm_: " << dsm_->bsm_;
    //LOG(INFO) << "Hold DSM: DONE";
}

void SyncedMemoryGuard_Content_Dynamic::release_owner( ) {
	if (!dsm_) return;
    boost::mutex::scoped_lock lock(dsm_->m_);
    auto iter = dsm_->guard_set_.find(this);
    CHECK(iter!=dsm_->guard_set_.end()) <<
    		"Internal error: orphan SyncedMemoryGuard detected";
	dsm_->guard_set_.erase(iter);
	if (dsm_->guard_set_.empty()) {
		//LOG(INFO) << "Release: " << dsm_->bsm_;
		dsm_->bsm_ = NULL;
		dsm_->bsm_holder_.reset();
	}
    dsm_=NULL;
}


boost::mutex SyncedMemoryAllocator::generic_aux::mu;

SyncedMemoryAllocator& SyncedMemoryAllocator::numbered_allocator( int id ) {
    boost::mutex::scoped_lock lock( generic_aux::mu );
	static std::map<int,SyncedMemoryAllocator> mdsma;
	return mdsma[id];
}


}  // namespace caffe

