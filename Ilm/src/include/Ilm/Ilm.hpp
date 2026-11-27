/**************************************************************************
 * Intrynzic Lifetime Manager: https://github.com/intrynzic/Ilm
 *
 * A single-header library implemented in C++26 for managing the lifetimes
 * and ownership semantics of memory resources.
 * ------------------------------------------------------------------------
 *
 * Copyright 2026 Intrynzic Software
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http ://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/

#pragma once

#ifdef ILM_BUILD_MODULE
#    define ILM_EXPORT export
import std;
#else
#    define ILM_EXPORT
#    include <atomic>
#    include <concepts>
#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <ostream>
#    include <type_traits>
#    include <utility>
#endif

#ifndef ILM_OMIT_NAMESPACE
#    define ILM_NAMESPACE_BEGIN  \
        ILM_EXPORT namespace ilm \
        {
#    define ILM_NAMESPACE_END }
#    define _ILM              ::ilm::
#else
#    define ILM_NAMESPACE_BEGIN
#    define ILM_NAMESPACE_END
#    define _ILM
#endif  // !ILM_OMIT_NAMESPACE

ILM_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// Internal constraints
// ---------------------------------------------------------------------------

// T2 is the same class as, or is derived from, Base.
// Deliberately identical to the old std::is_base_of_v<Base, T2> constraint
// (NOTE: not std::derived_from, which additionally demands public, unambiguous inheritance).
// Usage: template<_DerivedFrom<T> T2> means T2 must derive from T.
template <typename T2, typename Base>
concept _DerivedFrom = std::is_base_of_v<Base, T2>;

// Gates operator-> / operator* so that Ref<void> / Scope<void> remain usable.
template <typename T>
concept _NonVoid = !std::is_void_v<T>;

// Gates CreateScope / CreateRef (array types are not supported yet).
// TODO(Adam): Add support for arrays!
template <typename T>
concept _NonArray = !std::is_array_v<T>;

// Base is T2 with const added. Only for non-class types such as int, because std::is_base_of is
// always false for those so _DerivedFrom cannot express it (class types are covered by _DerivedFrom).
// Usage: template<_AddsConst<T> T2> means T must be const T2
template <typename T2, typename Base>
concept _AddsConst = !std::is_class_v<T2> && !std::is_const_v<T2> && std::is_same_v<const T2, Base>;

ILM_EXPORT template <typename T>
class Scope
{
public:
    constexpr explicit Scope(T* ptr) noexcept
        : m_Ptr(ptr), m_Deleter(&_DefaultDelete<T>) { };

    template <_DerivedFrom<T> T2>
    constexpr explicit Scope(T2* ptr) noexcept
        : m_Ptr(ptr), m_Deleter(&_DefaultDelete<T2>) { };

    template <_DerivedFrom<T> T2>
    constexpr Scope(Scope<T2>&& scope) noexcept
        : m_Ptr(scope.Release()), m_Deleter(std::exchange(scope.m_Deleter, nullptr)) { };

    constexpr Scope(Scope<T>&& scope) noexcept
        : m_Ptr(scope.Release()), m_Deleter(std::exchange(scope.m_Deleter, nullptr)) { };

    Scope(Scope<T>&)       = delete;
    Scope(const Scope<T>&) = delete;

    constexpr Scope(std::nullptr_t) noexcept
        : m_Ptr(nullptr), m_Deleter(nullptr) { };

    constexpr Scope() noexcept = default;

    constexpr ~Scope() noexcept
    {
        if (m_Ptr)
        {
            if (m_Deleter)
                m_Deleter((void*)m_Ptr);

            m_Ptr = nullptr;
        }
    }

    constexpr void Swap(Scope<T>& other) noexcept
    {
        if (this != &other)
        {
            std::swap(m_Ptr, other.m_Ptr);
            std::swap(m_Deleter, other.m_Deleter);
        }
    }

    template <_DerivedFrom<T> T2>
    constexpr void Reset(T2* newPtr) noexcept
    {
        Scope<T2>(newPtr).Swap(*this);
    }

    constexpr void Reset(T* newPtr) noexcept { Scope<T>(newPtr).Swap(*this); }

    constexpr void Reset() noexcept { Scope<T>(nullptr).Swap(*this); }

    constexpr T* Release() noexcept { return std::exchange(m_Ptr, nullptr); }

    constexpr T* Raw() const noexcept { return m_Ptr; }

    constexpr bool Valid() const noexcept { return Raw() != nullptr; }

    constexpr explicit operator bool() const noexcept { return Valid(); }

    Scope<T>& operator=(Scope<T>&)       = delete;
    Scope<T>& operator=(const Scope<T>&) = delete;

    constexpr Scope<T>& operator=(Scope<T>&& other) noexcept
    {
        Scope<T>(std::move(other)).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr Scope<T>& operator=(Scope<T2>&& other) noexcept
    {
        Scope<T>(std::move(other)).Swap(*this);
        return *this;
    }

    constexpr Scope<T>& operator=(std::nullptr_t) noexcept
    {
        Scope<T>(nullptr).Swap(*this);
        return *this;
    }

    constexpr T* operator->() const noexcept
        requires _NonVoid<T>
    {
        return Raw();
    }

    constexpr std::add_lvalue_reference_t<T> operator*() const noexcept
        requires _NonVoid<T>
    {
        return *Raw();
    }

private:
    template <typename T2>
    static void _DefaultDelete(void* ptr) noexcept
    {
        delete static_cast<T2*>(ptr);
    }

private:
    template <typename T2>
    friend class Scope;

private:
    T* m_Ptr                 = nullptr;
    void (*m_Deleter)(void*) = nullptr;  // Type-erased deleter
};

ILM_EXPORT template <_NonArray T, typename... Args>
constexpr Scope<T> CreateScope(Args&&... args) noexcept
{
    return Scope<T>(new T(std::forward<Args>(args)...));
}

ILM_EXPORT template <typename Wanted, typename T>
Wanted* GetScopeBaseTypePtr(const Scope<T>& scope) noexcept
{
    return static_cast<Wanted*>(scope.Raw());
}

class _AtomicRefCount
{
public:
    constexpr _AtomicRefCount() noexcept  = default;
    constexpr ~_AtomicRefCount() noexcept = default;

    _AtomicRefCount(const _AtomicRefCount&)            = delete;
    _AtomicRefCount& operator=(const _AtomicRefCount&) = delete;

    uint32_t GetStrongs() const noexcept { return m_Strongs.load(std::memory_order_acquire); }

    uint32_t GetWeaks() const noexcept { return m_Weaks.load(std::memory_order_acquire); }

    uint32_t IncRef() noexcept { return m_Strongs.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t DecRef() noexcept { return m_Strongs.fetch_sub(1, std::memory_order_acq_rel) - 1; }

    uint32_t IncWeakRef() noexcept { return m_Weaks.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t DecWeakRef() noexcept { return m_Weaks.fetch_sub(1, std::memory_order_acq_rel) - 1; }

private:
    std::atomic_uint m_Strongs = 1;  // TODO(Adam): We can highkey use bitset here
    std::atomic_uint m_Weaks   = 0;
};

template <typename T>
class Ref;

template <typename T>
class WeakRef;

// Base class for Ref, UnsafeRef and WeakRef
// std::remove_extent<> will need to be used in future to support array types.
template <typename T>
class _RefBase
{
protected:
    constexpr _RefBase(std::nullptr_t) noexcept
        : m_Ptr(nullptr), m_RefCount(nullptr), m_Deleter(nullptr) { };

    constexpr _RefBase() noexcept  = default;
    constexpr ~_RefBase() noexcept = default;

public:
    _RefBase(const _RefBase<T>&)               = delete;
    _RefBase<T>& operator=(const _RefBase<T>&) = delete;

protected:
    constexpr T* _Raw() const noexcept { return m_Ptr; }

    constexpr T* _ReleaseRaw() noexcept { return std::exchange(m_Ptr, nullptr); }

    constexpr void _Swap(_RefBase<T>& other) noexcept
    {
        std::swap(m_Ptr, other.m_Ptr);
        std::swap(m_RefCount, other.m_RefCount);
        std::swap(m_Deleter, other.m_Deleter);
    }

    uint32_t _RefCount() const noexcept { return m_RefCount ? m_RefCount->GetStrongs() : 0; }

    uint32_t _WeakRefCount() const noexcept { return m_RefCount ? m_RefCount->GetWeaks() : 0; }

    void _IncRef() noexcept
    {
        if (m_RefCount)
            (void)m_RefCount->IncRef();
    }

    void _DecRef() noexcept
    {
        if (m_RefCount && (m_RefCount->DecRef() == 0))
        {
            if (m_Ptr)
            {
                if (m_Deleter)
                    m_Deleter((void*)m_Ptr);

                m_Ptr = nullptr;
            }

            if (_WeakRefCount() == 0)
            {
                delete m_RefCount;
                m_RefCount = nullptr;
            }
        }
    }

    void _IncWeakRef() noexcept
    {
        if (m_RefCount)
            (void)m_RefCount->IncWeakRef();
    }

    void _DecWeakRef() noexcept
    {
        if (m_RefCount && (m_RefCount->DecWeakRef() == 0) && (_RefCount() == 0))
        {
            delete m_RefCount;
            m_RefCount = nullptr;
        }
    }

    template <typename T2>
    constexpr void _ConstructFromRaw(T2* ptr) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr);
        m_RefCount = m_Ptr ? new _AtomicRefCount() : nullptr;
        m_Deleter  = m_Ptr ? &_DefaultDelete<T2> : nullptr;
    }

    template <typename T2, typename T3>
    constexpr void _AliasConstructFrom(T2* ptr, const Ref<T3>& ref) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr);
        m_RefCount = ref.m_RefCount;
        m_Deleter  = ref.m_Deleter;

        // Guard against incrementing if we alias-constructed from an empty ref
        if (m_RefCount)
            _IncRef();
    }

    template <typename T2, typename T3>
    constexpr void _MoveAliasConstructFrom(T2* ptr, Ref<T3>&& ref) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr);
        m_RefCount = ref.m_RefCount;
        m_Deleter  = ref.m_Deleter;

        ref.m_Ptr      = nullptr;
        ref.m_RefCount = nullptr;
        ref.m_Deleter  = nullptr;
    }

    template <typename T2>
    constexpr void _MoveConstructFrom(_RefBase<T2>&& ptr) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr.m_Ptr);
        m_RefCount = ptr.m_RefCount;
        m_Deleter  = ptr.m_Deleter;

        ptr.m_Ptr      = nullptr;
        ptr.m_RefCount = nullptr;
        ptr.m_Deleter  = nullptr;
    }

    template <typename T2>
    constexpr void _CopyConstructFrom(const Ref<T2>& ref) noexcept
    {
        m_Ptr      = static_cast<T*>(ref.m_Ptr);
        m_RefCount = ref.m_RefCount;
        m_Deleter  = ref.m_Deleter;

        _IncRef();
    }

    template <typename T2>
    constexpr void _WeaklyConstructFrom(const _RefBase<T2>& ptr) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr.m_Ptr);
        m_RefCount = ptr.m_RefCount;
        m_Deleter  = ptr.m_Deleter;

        _IncWeakRef();
    }

    template <typename T2>
    constexpr void _ConstructFromWeak(const WeakRef<T2>& weak) noexcept
    {
        // We cannot create a strong reference to the resource if it was already deleted
        if (weak._RefCount() == 0)
        {
            m_Ptr      = nullptr;
            m_RefCount = nullptr;
            m_Deleter  = nullptr;
            return;
        }

        m_Ptr      = static_cast<T*>(weak.m_Ptr);
        m_RefCount = weak.m_RefCount;
        m_Deleter  = weak.m_Deleter;

        _IncRef();
    }

    template <typename T2>
    constexpr void _UnsafelyConstructFrom(const _RefBase<T2>& ptr) noexcept
    {
        m_Ptr      = static_cast<T*>(ptr.m_Ptr);
        m_RefCount = ptr.m_RefCount;
        m_Deleter  = ptr.m_Deleter;
    }

private:
    template <typename T2>
    static void _DefaultDelete(void* ptr) noexcept
    {
        delete static_cast<T2*>(ptr);
    }

private:
    T* m_Ptr                    = nullptr;
    _AtomicRefCount* m_RefCount = nullptr;
    void (*m_Deleter)(void*)    = nullptr;  // Type-erased deleter

private:
    template <typename T2>
    friend class _RefBase;

    friend class Ref<T>;
    friend class WeakRef<T>;
};

ILM_EXPORT template <typename T>
class Ref : public _RefBase<T>
{
public:
    constexpr explicit Ref(T* ptr) noexcept { this->_ConstructFromRaw(ptr); }

    template <_DerivedFrom<T> T2>
    constexpr explicit Ref(T2* ptr) noexcept
    {
        this->_ConstructFromRaw(ptr);
    }

    template <_DerivedFrom<T> T2>
    constexpr Ref(const Ref<T2>& ref) noexcept
    {
        this->_CopyConstructFrom(ref);
    }

    template <_AddsConst<T> T2>
    constexpr Ref(const Ref<T2>& ref) noexcept
    {
        this->_CopyConstructFrom(ref);
    }

    constexpr Ref(const Ref<T>& ref) noexcept { this->_CopyConstructFrom(ref); }

    template <typename T2, typename T3>
    constexpr Ref(T2* newPtr, const Ref<T3>& owner) noexcept
    {
        this->_AliasConstructFrom(newPtr, owner);
    }

    template <typename T2, typename T3>
    constexpr Ref(T2* newPtr, Ref<T3>&& owner) noexcept
    {
        this->_MoveAliasConstructFrom(newPtr, std::move(owner));
    }

    template <_DerivedFrom<T> T2>
    constexpr Ref(Ref<T2>&& ref) noexcept
    {
        this->_MoveConstructFrom(std::move(ref));
    }

    constexpr Ref(Ref<T>&& ref) noexcept { this->_MoveConstructFrom(std::move(ref)); }

    template <_DerivedFrom<T> T2>
    constexpr explicit Ref(const WeakRef<T2>& weak) noexcept
    {
        this->_ConstructFromWeak(weak);
    }

    constexpr explicit Ref(const WeakRef<T>& weak) noexcept { this->_ConstructFromWeak(weak); }

    constexpr Ref(std::nullptr_t) noexcept
        : _RefBase<T>(nullptr) { };

    constexpr Ref() noexcept = default;
    constexpr ~Ref() noexcept { this->_DecRef(); }

    constexpr void Swap(Ref<T>& other) noexcept
    {
        if (this != &other)
            this->_Swap(other);
    }

    template <_DerivedFrom<T> T2>
    constexpr void Reset(T2* newPtr) noexcept
    {
        Ref<T2>(newPtr).Swap(*this);
    }

    constexpr void Reset(T* newPtr) noexcept { Ref<T>(newPtr).Swap(*this); }

    constexpr void Reset() noexcept { Ref<T>(nullptr).Swap(*this); }

    constexpr T* Release() noexcept
    {
        T* res = this->_ReleaseRaw();
        Ref<T>(nullptr).Swap(*this);

        return res;
    }

    uint32_t RefCount() const noexcept { return this->_RefCount(); }

    bool Unique() const noexcept { return RefCount() == 1; }

    constexpr T* Raw() const noexcept { return this->_Raw(); }

    constexpr bool Valid() const noexcept { return Raw() != nullptr; }

    constexpr explicit operator bool() const noexcept { return Valid(); }

    Ref<T>& operator=(const Ref<T>& other) noexcept
    {
        Ref<T>(other).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr Ref<T>& operator=(const Ref<T2>& other) noexcept
    {
        Ref<T>(other).Swap(*this);
        return *this;
    }

    Ref<T>& operator=(Ref<T>&& other) noexcept
    {
        Ref<T>(std::move(other)).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr Ref<T>& operator=(Ref<T2>&& other) noexcept
    {
        Ref<T>(std::move(other)).Swap(*this);
        return *this;
    }

    constexpr Ref<T>& operator=(std::nullptr_t) noexcept
    {
        Ref<T>(nullptr).Swap(*this);
        return *this;
    }

    constexpr T* operator->() const noexcept
        requires _NonVoid<T>
    {
        return this->Raw();
    }

    constexpr std::add_lvalue_reference_t<T> operator*() const noexcept
        requires _NonVoid<T>
    {
        return *Raw();
    }

private:
    template <typename T2>
    friend class Ref;
};

ILM_EXPORT template <_NonArray T, typename... Args>
constexpr Ref<T> CreateRef(Args&&... args) noexcept
{
    return Ref<T>(new T(std::forward<Args>(args)...));
}

ILM_EXPORT template <typename Wanted, typename T>
Wanted* GetRefBaseTypePtr(const Ref<T>& ref) noexcept
{
    return static_cast<Wanted*>(ref.Raw());
}

// A non-owning view that grants temporary direct access to reference count
// manipulation for an existing Ref.
//
// This type does not participate in ownership or lifetime management. It may be
// used to increment or decrement the underlying object's reference count when
// higher-level ownership semantics are not suitable.
//
// Important notes:
//  * The referenced object must remain valid for the entire lifetime of this
//    instance. This class makes no guarantees about lifetime safety.
//  * Misuse can cause leaks, premature destruction, or inconsistencies in
//    ownership models. It should only be used when a clear manual strategy for
//    reference handling exists.
//  * Constructing this type is an explicit opt-in to bypassing automatic
//    lifetime management. Only cast the magic unsafe spell if you know what you're doing!
//
// Use with extreme caution, for this type is dangerous...
ILM_EXPORT template <typename T>
class UnsafeRef : public _RefBase<T>
{
public:
    constexpr UnsafeRef(const Ref<T>& ref) noexcept { this->_UnsafelyConstructFrom(ref); }

    template <_DerivedFrom<T> T2>
    constexpr UnsafeRef(const Ref<T2>& ref) noexcept
    {
        this->_UnsafelyConstructFrom(ref);
    }

    constexpr UnsafeRef(std::nullptr_t) noexcept
        : _RefBase<T>(nullptr) { };

    constexpr UnsafeRef()  = default;
    constexpr ~UnsafeRef() = default;

    constexpr void Swap(UnsafeRef<T>& other) noexcept
    {
        if (this != &other)
            this->_Swap(other);
    }

    void IncRef() noexcept { this->_IncRef(); }
    void DecRef() noexcept { this->_DecRef(); }

    void IncWeakRef() noexcept { this->_IncWeakRef(); }
    void DecWeakRef() noexcept { this->_DecWeakRef(); }

    uint32_t RefCount() const noexcept { return this->_RefCount(); }

    uint32_t WeakRefCount() const noexcept { return this->_WeakRefCount(); }

    constexpr bool Valid() const noexcept { return this->_Raw() != nullptr; }

    constexpr explicit operator bool() const noexcept { return Valid(); }

    constexpr UnsafeRef<T>& operator=(std::nullptr_t) noexcept
    {
        UnsafeRef<T>(nullptr).Swap(*this);
        return *this;
    }

    // NOTE: To prevent accidental usages, we intentionally deleted the Ref assignment operator overloads.
    constexpr UnsafeRef<T>& operator=(const Ref<T>&) noexcept = delete;
    constexpr UnsafeRef<T>& operator=(Ref<T>&&) noexcept      = delete;

private:
    template <typename T2>
    friend class UnsafeRef;
};

// Unsafe cast and copy
ILM_EXPORT template <typename T>
UnsafeRef<T> UnsafeRefCast(const Ref<T>& ref) noexcept
{
    return UnsafeRef<T>(ref);
}

// Static cast and copy
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> StaticRefCast(const Ref<T>& ref) noexcept
{
    const auto ptr = static_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, ref);
}

// Static cast and move
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> StaticRefCast(Ref<T>&& ref) noexcept
{
    const auto ptr = static_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, std::move(ref));
}

// Dynamic cast and copy
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> DynamicRefCast(const Ref<T>& ref) noexcept
{
    const auto ptr = dynamic_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, ref);
}

// Dynamic cast and move
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> DynamicRefCast(Ref<T>&& ref) noexcept
{
    const auto ptr = dynamic_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, std::move(ref));
}

// Reinterpret cast and copy
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> ReinterpretRefCast(const Ref<T>& ref) noexcept
{
    const auto ptr = reinterpret_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, ref);
}

// Reinterpret cast and move
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> ReinterpretRefCast(Ref<T>&& ref) noexcept
{
    const auto ptr = reinterpret_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, std::move(ref));
}

// Const cast and copy
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> ConstRefCast(const Ref<T>& ref) noexcept
{
    const auto ptr = const_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, ref);
}

// Const cast and move
ILM_EXPORT template <typename Wanted, typename T>
Ref<Wanted> ConstRefCast(Ref<T>&& ref) noexcept
{
    const auto ptr = const_cast<Wanted*>(ref.Raw());
    return Ref<Wanted>(ptr, std::move(ref));
}

ILM_EXPORT template <typename T>
class WeakRef : public _RefBase<T>
{
public:
    constexpr WeakRef(const Ref<T>& ref) noexcept { this->_WeaklyConstructFrom(ref); }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef(const Ref<T2>& ref) noexcept
    {
        this->_WeaklyConstructFrom(ref);
    }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef(const WeakRef<T2>& weakRef) noexcept
    {
        this->_WeaklyConstructFrom(weakRef);
    }

    template <_AddsConst<T> T2>
    constexpr WeakRef(const WeakRef<T2>& weakRef) noexcept
    {
        this->_WeaklyConstructFrom(weakRef);
    }

    constexpr WeakRef(const WeakRef<T>& weakRef) noexcept { this->_WeaklyConstructFrom(weakRef); }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef(WeakRef<T2>&& weakRef) noexcept
    {
        this->_MoveConstructFrom(std::move(weakRef));
    }

    constexpr WeakRef(WeakRef<T>&& weakRef) noexcept { this->_MoveConstructFrom(std::move(weakRef)); }

    constexpr WeakRef(std::nullptr_t) noexcept
        : _RefBase<T>(nullptr) { };

    constexpr WeakRef() noexcept = default;
    constexpr ~WeakRef() noexcept { this->_DecWeakRef(); }

    constexpr void Swap(WeakRef<T>& other) noexcept
    {
        if (this != &other)
            this->_Swap(other);
    }

    constexpr void Reset() noexcept { WeakRef<T>(nullptr).Swap(*this); }

    uint32_t RefCount() const noexcept { return this->_RefCount(); }

    bool Unique() const noexcept { return RefCount() == 1; }

    bool Expired() const noexcept { return RefCount() == 0; }

    constexpr T* Raw() const noexcept { return this->_Raw(); }

    constexpr bool Valid() const noexcept { return Raw() != nullptr; }

    Ref<T> Lock() const noexcept { return Expired() ? nullptr : Ref<T>(*this); }

    constexpr explicit operator bool() const noexcept { return Valid(); }

    WeakRef<T>& operator=(const WeakRef<T>& other) noexcept
    {
        WeakRef<T>(other).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef<T>& operator=(const WeakRef<T2>& other) noexcept
    {
        WeakRef<T>(other).Swap(*this);
        return *this;
    }

    WeakRef<T>& operator=(WeakRef<T>&& other) noexcept
    {
        WeakRef<T>(std::move(other)).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef<T>& operator=(WeakRef<T2>&& other) noexcept
    {
        WeakRef<T>(std::move(other)).Swap(*this);
        return *this;
    }

    WeakRef<T>& operator=(const Ref<T>& ref) noexcept
    {
        WeakRef<T>(ref).Swap(*this);
        return *this;
    }

    template <_DerivedFrom<T> T2>
    constexpr WeakRef<T>& operator=(const Ref<T2>& ref) noexcept
    {
        WeakRef<T>(ref).Swap(*this);
        return *this;
    }

    constexpr WeakRef<T>& operator=(std::nullptr_t) noexcept
    {
        WeakRef<T>(nullptr).Swap(*this);
        return *this;
    }

private:
    template <typename T2>
    friend class WeakRef;
};

ILM_EXPORT template <typename _Elem, typename _Traits, typename T>
std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& ostream, const Scope<T>& ptr)
{
    return ostream << ptr.Raw();
}

ILM_EXPORT template <typename _Elem, typename _Traits, typename T>
std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& ostream, const Ref<T>& ptr)
{
    return ostream << ptr.Raw();
}

ILM_EXPORT template <typename _Elem, typename _Traits, typename T>
std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& ostream, const WeakRef<T>& ptr)
{
    return ostream << ptr.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() == right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() == right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(T1* left, const Ref<T2> right) noexcept
{
    return left == right.Raw();
}

ILM_EXPORT template <typename T>
bool operator==(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() == nullptr;
}

ILM_EXPORT template <typename T>
bool operator==(std::nullptr_t, const Ref<T>& right) noexcept
{
    return nullptr == right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() != right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() != right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(T1* left, const Ref<T2>& right) noexcept
{
    return left != right.Raw();
}

ILM_EXPORT template <typename T>
bool operator!=(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() != nullptr;
}

ILM_EXPORT template <typename T>
bool operator!=(std::nullptr_t, const Ref<T>& right) noexcept
{
    return nullptr != right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() < right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() < right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(T1* left, const Ref<T2>& right) noexcept
{
    return left < right.Raw();
}

ILM_EXPORT template <typename T>
bool operator<(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() < static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator<(std::nullptr_t, const Ref<T>& right) noexcept
{
    return static_cast<T*>(nullptr) < right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() <= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() <= right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(T1* left, const Ref<T2>& right) noexcept
{
    return left <= right.Raw();
}

ILM_EXPORT template <typename T>
bool operator<=(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() <= static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator<=(std::nullptr_t, const Ref<T>& right) noexcept
{
    return static_cast<T*>(nullptr) <= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() > right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() > right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(T1* left, const Ref<T2>& right) noexcept
{
    return left > right.Raw();
}

ILM_EXPORT template <typename T>
bool operator>(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() > static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator>(std::nullptr_t, const Ref<T>& right) noexcept
{
    return static_cast<T*>(nullptr) > right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(const Ref<T1>& left, const Ref<T2>& right) noexcept
{
    return left.Raw() >= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(const Ref<T1>& left, T2* right) noexcept
{
    return left.Raw() >= right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(T1* left, const Ref<T2>& right) noexcept
{
    return left >= right.Raw();
}

ILM_EXPORT template <typename T>
bool operator>=(const Ref<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() >= static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator>=(std::nullptr_t, const Ref<T>& right) noexcept
{
    return static_cast<T*>(nullptr) >= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(const WeakRef<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() == right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() == right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator==(T1* left, const WeakRef<T2> right) noexcept
{
    return left == right.Raw();
}

ILM_EXPORT template <typename T>
bool operator==(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() == nullptr;
}

ILM_EXPORT template <typename T>
bool operator==(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return nullptr == right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(const WeakRef<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() != right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() != right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator!=(T1* left, const WeakRef<T2>& right) noexcept
{
    return left != right.Raw();
}

ILM_EXPORT template <typename T>
bool operator!=(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() != nullptr;
}

ILM_EXPORT template <typename T>
bool operator!=(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return nullptr != right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(const Ref<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() < right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() < right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<(T1* left, const WeakRef<T2>& right) noexcept
{
    return left < right.Raw();
}

ILM_EXPORT template <typename T>
bool operator<(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() < static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator<(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return static_cast<T*>(nullptr) < right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(const WeakRef<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() <= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() <= right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator<=(T1* left, const WeakRef<T2>& right) noexcept
{
    return left <= right.Raw();
}

ILM_EXPORT template <typename T>
bool operator<=(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() <= static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator<=(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return static_cast<T*>(nullptr) <= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(const WeakRef<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() > right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() > right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>(T1* left, const WeakRef<T2>& right) noexcept
{
    return left > right.Raw();
}

ILM_EXPORT template <typename T>
bool operator>(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() > static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator>(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return static_cast<T*>(nullptr) > right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(const WeakRef<T1>& left, const WeakRef<T2>& right) noexcept
{
    return left.Raw() >= right.Raw();
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(const WeakRef<T1>& left, T2* right) noexcept
{
    return left.Raw() >= right;
}

ILM_EXPORT template <typename T1, typename T2>
bool operator>=(T1* left, const WeakRef<T2>& right) noexcept
{
    return left >= right.Raw();
}

ILM_EXPORT template <typename T>
bool operator>=(const WeakRef<T>& left, std::nullptr_t) noexcept
{
    return left.Raw() >= static_cast<T*>(nullptr);
}

ILM_EXPORT template <typename T>
bool operator>=(std::nullptr_t, const WeakRef<T>& right) noexcept
{
    return static_cast<T*>(nullptr) >= right.Raw();
}

ILM_NAMESPACE_END

namespace std
{
    template <class _Kty>
    struct hash;

    ILM_EXPORT template <typename T>
    struct hash<_ILM Scope<T>>
    {
        size_t operator()(const _ILM Scope<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };

    ILM_EXPORT template <typename T>
    struct hash<_ILM Ref<T>>
    {
        size_t operator()(const _ILM Ref<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };

    ILM_EXPORT template <typename T>
    struct hash<_ILM WeakRef<T>>
    {
        size_t operator()(const _ILM WeakRef<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };
}  // namespace std
