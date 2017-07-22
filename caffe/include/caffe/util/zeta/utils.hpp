#ifndef ZETA_UTILS_HPP_
#define ZETA_UTILS_HPP_

namespace zeta {

template <typename FST>
inline FST& get_first_arg( FST& first ... ){
    return first;
}

template <typename> struct matlab_type {
    static const char* name() { return "void"; };
};
template<> struct matlab_type<float> {
    static const char* name() { return "single"; };
};
template<> struct matlab_type<double> {
    static const char* name() { return "double"; };
};
}

#endif
