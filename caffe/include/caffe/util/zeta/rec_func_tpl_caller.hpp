/*
 * rec_func_tpl_caller.hpp
 *
 *  Created on: Aug 14, 2015
 *      Author: zhangyuting
 */

#ifndef REC_FUNC_TPL_CALLER_HPP_
#define REC_FUNC_TPL_CALLER_HPP_

#include <exception>
#include <type_traits>

namespace zeta {

class rec_func_tpl_call_overflow : public std::exception {
public:
	virtual ~rec_func_tpl_call_overflow() throw() {}
	virtual const char* what() const throw() {
		return "rec_func_tpl_call_overflow";
	}
};

template<int m, template<int k> class func_tpl, class ...Args>
class rec_func_tpl_caller {
public:
	typedef func_tpl<m> func_t;
	typedef typename std::result_of<func_t(Args...)>::type R;
	R operator() ( int n, Args ...args ) const {
		if ( n > m ) {
			return rec_func_tpl_caller<m+1, func_tpl, Args... >()( n, args... );
		} else if ( n == m ) {
			return func_t()(args...);
		} else {
			throw rec_func_tpl_call_overflow();
		}
	}
};

template<template<int k> class func_tpl, class ...Args>
class rec_func_tpl_caller<0,func_tpl,Args...> {
public:
	typedef func_tpl<0> func_t;
	typedef typename std::result_of<func_t(Args...)>::type R;
	R operator() ( int n, Args ...args ) const {
		if ( n ) throw rec_func_tpl_call_overflow();
		else return func_t()(args...);
	}
};

template<int maxN, template<int k> class func_tpl0 >
class rec_func_tpl_offset {
public:
	template<int m> using func_tpl = func_tpl0<m+maxN>;
};

template<int minN, int maxN, template<int k> class func_tpl0, class ...Args>
class rec_func_tpl_caller_func_t {
public:
    typedef func_tpl0<maxN> func_t;
    typedef typename std::result_of< func_t(Args...) >::type return_type;
    return_type operator() ( int n, Args ...args ) const {
        typedef rec_func_tpl_offset<maxN, func_tpl0 > Func_T;
	    typedef rec_func_tpl_caller< minN-maxN, Func_T::template func_tpl, Args... > c;
	    return c()(n-maxN, args...);
    }
};

template<int minN, int maxN, template<int k> class func_tpl0, class ...Args>
typename rec_func_tpl_caller_func_t<minN,maxN,func_tpl0,Args...>::return_type
rec_func_tpl_caller_func( int n, Args ...args ) {
    return rec_func_tpl_caller_func_t<minN,maxN,func_tpl0,Args...>()(n, args...);
}

/*
#define WRAP_REC_FUNC_TPL( FUNC_TPL_NAME, WRAPPER_NAME ) \
	class WRAPPER_NAME { \
		template<int m> func_obj { public: decltype(FUNC_TPL_NAME<m>) operator()() {return FUNC_TPL_NAME<m>();} }; \
		template<int m> using func_tpl = FUNC_TPL_NAME<m>; \
	};
*/

#define WRAP_REC_FUNC_OBJ_TPL( FUNC_TPL_OBJ_NAME, WRAPPER_NAME ) \
	class WRAPPER_NAME { \
		template<int m> using func_tpl = FUNC_TPL_OBJ_NAME<m>; \
	};

}


#endif /* REC_FUNC_TPL_CALLER_HPP_ */
