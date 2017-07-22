#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <set>
#include <map>
#include <boost/lexical_cast.hpp>

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


class SyncedMemoryGuard_Content { };
typedef boost::shared_ptr<SyncedMemoryGuard_Content> SyncedMemoryGuard;

class BaseSyncedMemory {
 public:
  virtual ~BaseSyncedMemory() {}
  virtual const void* cpu_data() = 0;
  virtual void set_cpu_data(void* data) = 0;
  virtual const void* gpu_data() = 0;
  virtual void set_gpu_data(void* data) = 0;
  virtual void* mutable_cpu_data() = 0;
  virtual void* mutable_gpu_data() = 0;
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  virtual SyncedHead head() = 0;
  virtual size_t size() = 0;

#ifndef CPU_ONLY
virtual void async_gpu_push(const cudaStream_t& stream) = 0;
#endif

virtual SyncedMemoryGuard hold() { return SyncedMemoryGuard(); }
virtual bool is_held() { return true; }

//DISABLE_COPY_AND_ASSIGN(BaseSyncedMemory);
};  // class BaseSyncedMemory



/**
* @brief Manages memory allocation and synchronization between the host (CPU)
*        and device (GPU).
*
* TODO(dox): more thorough description.
*/
class StoredSyncedMemory : public BaseSyncedMemory {
public:
StoredSyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false), 
        gpu_device_(-1) {}
  explicit StoredSyncedMemory(size_t size)
	  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false), 
        gpu_device_(-1) {}
  virtual ~StoredSyncedMemory();
  virtual const void* cpu_data();
  virtual void set_cpu_data(void* data);
  virtual const void* gpu_data();
  virtual void set_gpu_data(void* data);
  virtual void* mutable_cpu_data();
  virtual void* mutable_gpu_data();
  virtual SyncedHead head() { return head_; }
  virtual size_t size() { return size_; }

#ifndef CPU_ONLY
  virtual void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(StoredSyncedMemory);
};  // class StoredSyncedMemory


class SyncedMemory : public BaseSyncedMemory {
 protected:
  boost::shared_ptr<BaseSyncedMemory> bsm_holder_;
  BaseSyncedMemory* bsm_;
 public:
  SyncedMemory() : bsm_holder_( (BaseSyncedMemory*) new StoredSyncedMemory ), bsm_(bsm_holder_.get()) {}
  SyncedMemory(size_t size) : bsm_holder_( (BaseSyncedMemory*) new StoredSyncedMemory(size) ), bsm_(bsm_holder_.get()) {}
  virtual ~SyncedMemory() {}
  virtual const void* cpu_data() { return bsm_->cpu_data(); }
  virtual void set_cpu_data(void* data) { return bsm_->set_cpu_data(data); }
  virtual const void* gpu_data() { return bsm_->gpu_data(); }
  virtual void set_gpu_data(void* data) { return bsm_->set_gpu_data(data); }
  virtual void* mutable_cpu_data() { return bsm_->mutable_cpu_data(); }
  virtual void* mutable_gpu_data() { return bsm_->mutable_gpu_data(); }
  virtual SyncedHead head() { return bsm_->head(); }
  virtual size_t size() { return bsm_->size(); }

#ifndef CPU_ONLY
  virtual void async_gpu_push(const cudaStream_t& stream) { return bsm_->async_gpu_push(stream); }
#endif
 
 protected:
  struct INIT_FROM_BASE{};
  SyncedMemory(const INIT_FROM_BASE&, BaseSyncedMemory& bsm) : bsm_( &bsm ) {}
  struct EMPTY_INIT{};
  SyncedMemory(const EMPTY_INIT&) : bsm_holder_(), bsm_(NULL) {}

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class StoredSyncedMemory



class OffsetSyncedMemory : public SyncedMemory {
 protected:
  using SyncedMemory::bsm_;
  size_t offset_;
 public:
  OffsetSyncedMemory( BaseSyncedMemory& bsm, size_t offset ) :
	  SyncedMemory(SyncedMemory::INIT_FROM_BASE(),bsm), offset_(offset) {}
  virtual ~OffsetSyncedMemory() {}
  virtual const void* cpu_data() { return (char*)bsm_->cpu_data() + offset_; }
  virtual const void* gpu_data() { return (char*)bsm_->gpu_data() + offset_; }
  virtual void* mutable_cpu_data() { return (char*)bsm_->mutable_cpu_data() + offset_; }
  virtual void* mutable_gpu_data() { return (char*)bsm_->mutable_gpu_data() + offset_; }
  virtual size_t size() { return bsm_->size() - offset_; }

  DISABLE_COPY_AND_ASSIGN(OffsetSyncedMemory);
};  // class OffsetSyncedMemory

// dynamic synced memory allocators ------------------------------------

typedef std::pair<boost::shared_ptr<SyncedMemory>, 
        boost::shared_ptr<boost::mutex> > synced_mem_with_mutex;

struct synced_mem_with_mutex_less {
	bool operator() ( const synced_mem_with_mutex& a, const synced_mem_with_mutex& b) {
		if (b.first.get()) {
			if (!a.first.get()) return true;
			return (a.first->size()<b.first->size());
		}
		return false;
	}
};

class LockedSyncedMemory : public SyncedMemory {
protected:
	boost::mutex::scoped_lock lock_;
public:
	LockedSyncedMemory( synced_mem_with_mutex smwm );
	bool is_locked() const { return bool(lock_); }
};

class SyncedMemoryAllocator {
public:
	typedef std::multiset<synced_mem_with_mutex,synced_mem_with_mutex_less> mem_bank_t;
private:
    shared_ptr<boost::mutex> m_;
	std::map<int,mem_bank_t> gb_;
public:
	boost::shared_ptr<LockedSyncedMemory> alloc( size_t s );

    struct generic_aux {
        static boost::mutex mu;
    };

	static SyncedMemoryAllocator& numbered_allocator( int id=0 );

    enum fixed_allocator_id {
    	GENERAL,
        CONV_COL,
        TINY,
        SMALL,
        MIDDLE,
        LARGE,
    };

    SyncedMemoryAllocator();

    template<int AllocId>
    static SyncedMemoryAllocator& fixed_allocator( );

public:
    const std::string tag;
    explicit SyncedMemoryAllocator(const std::string& tag_str);

};


template<int AllocId>
SyncedMemoryAllocator& SyncedMemoryAllocator::fixed_allocator( ) {
    boost::mutex::scoped_lock lock(generic_aux::mu);
    static SyncedMemoryAllocator fsma(boost::lexical_cast<std::string>(AllocId));
    return fsma;
}

class SyncedMemoryGuard_Content_Dynamic;

class DynamicSyncedMemory : public SyncedMemory {
	friend class SyncedMemoryGuard_Content_Dynamic;
protected:
    using SyncedMemory::bsm_holder_;	// use it for LockedSyncedMemory
    using SyncedMemory::bsm_;
    size_t size_;
    SyncedMemoryAllocator& sma_;
    std::set<SyncedMemoryGuard_Content_Dynamic*> guard_set_;
    boost::mutex m_;
public:
    DynamicSyncedMemory( size_t s, SyncedMemoryAllocator& sma =
    		SyncedMemoryAllocator::numbered_allocator() ) :
    	SyncedMemory(SyncedMemory::EMPTY_INIT()), size_(s) , sma_(sma) {}
    virtual ~DynamicSyncedMemory();
    virtual size_t size() {return size_;}
    virtual SyncedMemoryGuard hold();
    virtual bool is_held() { return !guard_set_.empty(); };

    virtual const void* cpu_data();
    virtual const void* gpu_data();
    virtual void* mutable_cpu_data();
    virtual void* mutable_gpu_data();

    virtual void transfer_guard_to( DynamicSyncedMemory& dsm );

};

class SyncedMemoryGuard_Content_Dynamic : public SyncedMemoryGuard_Content {
	friend class DynamicSyncedMemory;
protected:
	boost::mutex m_;
	DynamicSyncedMemory* dsm_;
	void change_owner( DynamicSyncedMemory& dsm );
	void set_owner( DynamicSyncedMemory& dsm );
	void release_owner( );
public:
	SyncedMemoryGuard_Content_Dynamic( DynamicSyncedMemory& dsm );
	~SyncedMemoryGuard_Content_Dynamic();
};

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
