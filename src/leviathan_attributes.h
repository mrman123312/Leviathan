#ifndef LEVIATHAN_ATTRIBUTES_H_INCLUDED
#define LEVIATHAN_ATTRIBUTES_H_INCLUDED

#if defined(_MSC_VER)
    #define LEVIATHAN_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define LEVIATHAN_NOINLINE __attribute__((noinline))
#else
    #define LEVIATHAN_NOINLINE
#endif

#endif  // LEVIATHAN_ATTRIBUTES_H_INCLUDED
