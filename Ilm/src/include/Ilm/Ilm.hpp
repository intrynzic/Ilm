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
 * https://www.apache.org/licenses/LICENSE-2.0
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
#    include <compare>
#    include <concepts>
#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <ostream>
#    include <type_traits>
#    include <utility>
#endif


namespace ilm
{
    // Forward declaration of Ref is required for detail::Refbase.
    template <typename T>
    class Ref;

    // Forward declaration of WeakRef is required for detail::Refbase.
    template <typename T>
    class WeakRef;

    // Contains internal implementation details.
    namespace detail
    {
        // T2 is the same class as, or is derived from, Base.
        // Deliberately identical to the old std::is_base_of_v<Base, T2> constraint
        // (NOTE: not std::derived_from, which additionally demands public, unambiguous inheritance).
        // Usage: template<detail::DerivedFrom<T> T2> means T2 must derive from T.
        template <typename T2, typename Base>
        concept DerivedFrom = std::is_base_of_v<Base, T2>;

        // Gates operator-> / operator* so that Ref<void> / Scope<void> remain usable.
        template <typename T>
        concept NonVoid = !std::same_as<T, void>;

        // Gates CreateScope / CreateRef (array types are not supported yet).
        // TODO(Adam): Add support for arrays!
        template <typename T>
        concept NonArray = !std::is_array_v<T>;

        // Base is T2 with const added. Only for non-class types such as int, because std::is_base_of is
        // always false for those so detail::DerivedFrom cannot express it (class types are covered by detail::DerivedFrom).
        // Usage: template<detail::AddsConst<T> T2> means T must be const T2
        template <typename T2, typename Base>
        concept AddsConst = !std::is_const_v<T2> && std::same_as<std::add_const_t<T2>, Base>;

        // Base case template for checking if a type is an instantiation of Ref/WeakRef.
        template <typename T>
        constexpr bool IsRefLike = false;

        // Ref<T> is an instantiation of Ref.
        template <typename T>
        constexpr bool IsRefLike<Ref<T>> = true;

        // WeakRef<T> is an instantiation of WeakRef which is RefLike.
        template <typename T>
        constexpr bool IsRefLike<WeakRef<T>> = true;

        // NOTE(Adam): We don't specialize this for UnsafeRef or provide comparison
        //             operator overloads as they should never be required!
        //             UnsafeRef should be hard and unappealing to use.

        // UnsafeRef<T> is an instantiation of UnsafeRef which is RefLike.
        // template <typename T>
        // constexpr inline bool IsRefLike<UnsafeRef<T>> = true;

        // T is wrapped into an instantiation of Ref, WeakRef or UnsafeRef.
        template <typename T>
        concept RefLike = IsRefLike<T>;

        // The control block that stores the atomic reference count.
        class AtomicRefCount
        {
        public:
            constexpr AtomicRefCount()  = default;
            constexpr ~AtomicRefCount() = default;

            AtomicRefCount(const AtomicRefCount&)            = delete;
            AtomicRefCount& operator=(const AtomicRefCount&) = delete;

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

        // Base class for Ref, UnsafeRef and WeakRef
        // std::remove_extent<> will need to be used in future to support array types.
        template <typename T>
        class RefBase
        {
        protected:
            constexpr RefBase(std::nullptr_t) noexcept
                : m_Ptr(nullptr), m_RefCount(nullptr), m_Deleter(nullptr) { };

            constexpr RefBase()  = default;
            constexpr ~RefBase() = default;

        public:
            RefBase(const RefBase<T>&)               = delete;
            RefBase<T>& operator=(const RefBase<T>&) = delete;

        protected:
            constexpr T* GetRaw() const noexcept { return m_Ptr; }

            [[nodiscard]] constexpr T* ReleaseImpl() noexcept { return std::exchange(m_Ptr, nullptr); }

            constexpr void SwapImpl(RefBase<T>& other) noexcept
            {
                std::swap(m_Ptr, other.m_Ptr);
                std::swap(m_RefCount, other.m_RefCount);
                std::swap(m_Deleter, other.m_Deleter);
            }

            uint32_t RefCountImpl() const noexcept { return m_RefCount ? m_RefCount->GetStrongs() : 0; }

            uint32_t WeakRefCountImpl() const noexcept { return m_RefCount ? m_RefCount->GetWeaks() : 0; }

            void IncRef() noexcept
            {
                if (m_RefCount)
                    (void)m_RefCount->IncRef();
            }

            void DecRef() noexcept
            {
                if (m_RefCount && (m_RefCount->DecRef() == 0))
                {
                    if (m_Ptr)
                    {
                        if (m_Deleter)
                            m_Deleter(static_cast<void*>(m_Ptr));

                        m_Ptr = nullptr;
                    }

                    if (WeakRefCountImpl() == 0)
                    {
                        delete m_RefCount;
                        m_RefCount = nullptr;
                    }
                }
            }

            void IncWeakRef() noexcept
            {
                if (m_RefCount)
                    (void)m_RefCount->IncWeakRef();
            }

            void DecWeakRef() noexcept
            {
                if (m_RefCount && (m_RefCount->DecWeakRef() == 0) && (RefCountImpl() == 0))
                {
                    delete m_RefCount;
                    m_RefCount = nullptr;
                }
            }

            template <typename T2>
            void ConstructFromRaw(T2* ptr) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr);
                m_RefCount = m_Ptr ? new AtomicRefCount() : nullptr;
                m_Deleter  = m_Ptr ? &DefaultDelete<T2> : nullptr;
            }

            template <typename T2, typename T3>
            constexpr void AliasConstructFrom(T2* ptr, const Ref<T3>& ref) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr);
                m_RefCount = ref.m_RefCount;
                m_Deleter  = ref.m_Deleter;

                // Guard against incrementing if we alias-constructed from an empty ref
                if (m_RefCount)
                    IncRef();
            }

            template <typename T2, typename T3>
            constexpr void MoveAliasConstructFrom(T2* ptr, Ref<T3>&& ref) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr);
                m_RefCount = ref.m_RefCount;
                m_Deleter  = ref.m_Deleter;

                ref.m_Ptr      = nullptr;
                ref.m_RefCount = nullptr;
                ref.m_Deleter  = nullptr;
            }

            template <typename T2>
            constexpr void MoveConstructFrom(RefBase<T2>&& ptr) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr.m_Ptr);
                m_RefCount = ptr.m_RefCount;
                m_Deleter  = ptr.m_Deleter;

                ptr.m_Ptr      = nullptr;
                ptr.m_RefCount = nullptr;
                ptr.m_Deleter  = nullptr;
            }

            template <typename T2>
            constexpr void CopyConstructFrom(const Ref<T2>& ref) noexcept
            {
                m_Ptr      = static_cast<T*>(ref.m_Ptr);
                m_RefCount = ref.m_RefCount;
                m_Deleter  = ref.m_Deleter;

                IncRef();
            }

            template <typename T2>
            constexpr void WeaklyConstructFrom(const RefBase<T2>& ptr) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr.m_Ptr);
                m_RefCount = ptr.m_RefCount;
                m_Deleter  = ptr.m_Deleter;

                IncWeakRef();
            }

            template <typename T2>
            constexpr void ConstructFromWeak(const WeakRef<T2>& weak) noexcept
            {
                // We cannot create a strong reference to the resource if it was already deleted
                if (weak.RefCountImpl() == 0)
                {
                    m_Ptr      = nullptr;
                    m_RefCount = nullptr;
                    m_Deleter  = nullptr;
                    return;
                }

                m_Ptr      = static_cast<T*>(weak.m_Ptr);
                m_RefCount = weak.m_RefCount;
                m_Deleter  = weak.m_Deleter;

                IncRef();
            }

            template <typename T2>
            constexpr void UnsafelyConstructFrom(const RefBase<T2>& ptr) noexcept
            {
                m_Ptr      = static_cast<T*>(ptr.m_Ptr);
                m_RefCount = ptr.m_RefCount;
                m_Deleter  = ptr.m_Deleter;
            }

        private:
            template <typename T2>
            static void DefaultDelete(void* ptr) noexcept
            {
                delete static_cast<T2*>(ptr);
            }

        private:
            T* m_Ptr                   = nullptr;
            AtomicRefCount* m_RefCount = nullptr;
            void (*m_Deleter)(void*)   = nullptr;  // Type-erased deleter

        private:
            template <typename T2>
            friend class RefBase;

            friend class Ref<T>;
            friend class WeakRef<T>;
        };
    }  // namespace detail

    ILM_EXPORT template <typename T>
    class Scope
    {
    public:
        constexpr explicit Scope(T* ptr) noexcept
            : m_Ptr(ptr), m_Deleter(&DefaultDelete<T>) { };

        template <detail::DerivedFrom<T> T2>
        constexpr explicit Scope(T2* ptr) noexcept
            : m_Ptr(ptr), m_Deleter(&DefaultDelete<T2>) { };

        template <detail::DerivedFrom<T> T2>
        constexpr Scope(Scope<T2>&& scope) noexcept
            : m_Ptr(scope.Release()), m_Deleter(std::exchange(scope.m_Deleter, nullptr)) { };

        constexpr Scope(Scope<T>&& scope) noexcept
            : m_Ptr(scope.Release()), m_Deleter(std::exchange(scope.m_Deleter, nullptr)) { };

        Scope(Scope<T>&)       = delete;
        Scope(const Scope<T>&) = delete;

        constexpr Scope(std::nullptr_t) noexcept
            : m_Ptr(nullptr), m_Deleter(nullptr) { };

        constexpr Scope() = default;

        constexpr ~Scope() noexcept
        {
            if (m_Ptr)
            {
                if (m_Deleter)
                    m_Deleter(static_cast<void*>(m_Ptr));

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

        template <detail::DerivedFrom<T> T2>
        constexpr void Reset(T2* newPtr) noexcept
        {
            Scope<T2>(newPtr).Swap(*this);
        }

        constexpr void Reset(T* newPtr) noexcept { Scope<T>(newPtr).Swap(*this); }

        constexpr void Reset() noexcept { Scope<T>(nullptr).Swap(*this); }

        [[nodiscard("Discarding a released resource leaks the resource!")]]
        constexpr T* Release() noexcept
        {
            return std::exchange(m_Ptr, nullptr);
        }

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

        template <detail::DerivedFrom<T> T2>
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
            requires detail::NonVoid<T>
        {
            return Raw();
        }

        constexpr std::add_lvalue_reference_t<T> operator*() const noexcept
            requires detail::NonVoid<T>
        {
            return *Raw();
        }

    private:
        template <typename T2>
        static void DefaultDelete(void* ptr) noexcept
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

    ILM_EXPORT template <detail::NonArray T, typename... Args>
    [[nodiscard("Discarding a Scope leaks the resource!")]]
    Scope<T> CreateScope(Args&&... args) noexcept
    {
        return Scope<T>(new T(std::forward<Args>(args)...));
    }

    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Wanted* GetScopeBaseTypePtr(const Scope<T>& scope) noexcept
    {
        return static_cast<Wanted*>(scope.Raw());
    }

    ILM_EXPORT template <typename T>
    class Ref : public detail::RefBase<T>
    {
    public:
        explicit Ref(T* ptr) noexcept { this->ConstructFromRaw(ptr); }

        template <detail::DerivedFrom<T> T2>
        explicit Ref(T2* ptr) noexcept
        {
            this->ConstructFromRaw(ptr);
        }

        template <detail::DerivedFrom<T> T2>
        constexpr Ref(const Ref<T2>& ref) noexcept
        {
            this->CopyConstructFrom(ref);
        }

        template <detail::AddsConst<T> T2>
        constexpr Ref(const Ref<T2>& ref) noexcept
        {
            this->CopyConstructFrom(ref);
        }

        constexpr Ref(const Ref<T>& ref) noexcept { this->CopyConstructFrom(ref); }

        template <typename T2, typename T3>
        constexpr Ref(T2* newPtr, const Ref<T3>& owner) noexcept
        {
            this->AliasConstructFrom(newPtr, owner);
        }

        template <typename T2, typename T3>
        constexpr Ref(T2* newPtr, Ref<T3>&& owner) noexcept
        {
            this->MoveAliasConstructFrom(newPtr, std::move(owner));
        }

        template <detail::DerivedFrom<T> T2>
        constexpr Ref(Ref<T2>&& ref) noexcept
        {
            this->MoveConstructFrom(std::move(ref));
        }

        constexpr Ref(Ref<T>&& ref) noexcept { this->MoveConstructFrom(std::move(ref)); }

        template <detail::DerivedFrom<T> T2>
        constexpr explicit Ref(const WeakRef<T2>& weak) noexcept
        {
            this->ConstructFromWeak(weak);
        }

        constexpr explicit Ref(const WeakRef<T>& weak) noexcept { this->ConstructFromWeak(weak); }

        constexpr Ref(std::nullptr_t) noexcept
            : detail::RefBase<T>(nullptr) { };

        constexpr Ref() = default;
        constexpr ~Ref() noexcept { this->DecRef(); }

        constexpr void Swap(Ref<T>& other) noexcept
        {
            if (this != &other)
                this->SwapImpl(other);
        }

        template <detail::DerivedFrom<T> T2>
        constexpr void Reset(T2* newPtr) noexcept
        {
            Ref<T2>(newPtr).Swap(*this);
        }

        constexpr void Reset(T* newPtr) noexcept { Ref<T>(newPtr).Swap(*this); }

        constexpr void Reset() noexcept { Ref<T>(nullptr).Swap(*this); }

        [[nodiscard("Discarding a released resource leaks the resource!")]]
        constexpr T* Release() noexcept
        {
            T* res = this->ReleaseImpl();
            Ref<T>(nullptr).Swap(*this);

            return res;
        }

        uint32_t RefCount() const noexcept { return this->RefCountImpl(); }

        bool Unique() const noexcept { return RefCount() == 1; }

        constexpr T* Raw() const noexcept { return this->GetRaw(); }

        constexpr bool Valid() const noexcept { return Raw() != nullptr; }

        constexpr explicit operator bool() const noexcept { return Valid(); }

        Ref<T>& operator=(const Ref<T>& other) noexcept
        {
            Ref<T>(other).Swap(*this);
            return *this;
        }

        template <detail::DerivedFrom<T> T2>
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

        template <detail::DerivedFrom<T> T2>
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
            requires detail::NonVoid<T>
        {
            return Raw();
        }

        constexpr std::add_lvalue_reference_t<T> operator*() const noexcept
            requires detail::NonVoid<T>
        {
            return *Raw();
        }

    private:
        template <typename T2>
        friend class Ref;
    };

    ILM_EXPORT template <detail::NonArray T, typename... Args>
    [[nodiscard("Discarding a Ref leaks the resource!")]]
    Ref<T> CreateRef(Args&&... args) noexcept
    {
        return Ref<T>(new T(std::forward<Args>(args)...));
    }

    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Wanted* GetRefBaseTypePtr(const Ref<T>& ref) noexcept
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
    class UnsafeRef : public detail::RefBase<T>
    {
    public:
        constexpr UnsafeRef(const Ref<T>& ref) noexcept { this->UnsafelyConstructFrom(ref); }

        template <detail::DerivedFrom<T> T2>
        constexpr UnsafeRef(const Ref<T2>& ref) noexcept
        {
            this->UnsafelyConstructFrom(ref);
        }

        constexpr UnsafeRef(std::nullptr_t) noexcept
            : detail::RefBase<T>(nullptr) { };

        constexpr UnsafeRef()  = default;
        constexpr ~UnsafeRef() = default;

        constexpr void Swap(UnsafeRef<T>& other) noexcept
        {
            if (this != &other)
                this->SwapImpl(other);
        }

        void IncRef() noexcept { this->IncRef(); }
        void DecRef() noexcept { this->DecRef(); }

        void IncWeakRef() noexcept { this->IncWeakRef(); }
        void DecWeakRef() noexcept { this->DecWeakRef(); }

        uint32_t RefCount() const noexcept { return this->RefCountImpl(); }

        uint32_t WeakRefCount() const noexcept { return this->WeakRefCountImpl(); }

        constexpr bool Valid() const noexcept { return this->Raw() != nullptr; }

        constexpr explicit operator bool() const noexcept { return Valid(); }

        constexpr UnsafeRef<T>& operator=(std::nullptr_t) noexcept
        {
            UnsafeRef<T>(nullptr).Swap(*this);
            return *this;
        }

        // NOTE: To prevent accidental usages, we intentionally deleted the Ref assignment operator overloads.
        constexpr UnsafeRef<T>& operator=(const Ref<T>&) = delete;
        constexpr UnsafeRef<T>& operator=(Ref<T>&&)      = delete;

    private:
        template <typename T2>
        friend class UnsafeRef;
    };

    // Unsafe cast and copy
    ILM_EXPORT template <typename T>
    [[nodiscard]] UnsafeRef<T> UnsafeRefCast(const Ref<T>& ref) noexcept
    {
        return UnsafeRef<T>(ref);
    }

    // Static cast and copy
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> StaticRefCast(const Ref<T>& ref) noexcept
    {
        const auto ptr = static_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, ref);
    }

    // Static cast and move
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> StaticRefCast(Ref<T>&& ref) noexcept
    {
        const auto ptr = static_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, std::move(ref));
    }

    // Dynamic cast and copy
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> DynamicRefCast(const Ref<T>& ref) noexcept
    {
        const auto ptr = dynamic_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, ref);
    }

    // Dynamic cast and move
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> DynamicRefCast(Ref<T>&& ref) noexcept
    {
        const auto ptr = dynamic_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, std::move(ref));
    }

    // Reinterpret cast and copy
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> ReinterpretRefCast(const Ref<T>& ref) noexcept
    {
        const auto ptr = reinterpret_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, ref);
    }

    // Reinterpret cast and move
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> ReinterpretRefCast(Ref<T>&& ref) noexcept
    {
        const auto ptr = reinterpret_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, std::move(ref));
    }

    // Const cast and copy
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> ConstRefCast(const Ref<T>& ref) noexcept
    {
        const auto ptr = const_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, ref);
    }

    // Const cast and move
    ILM_EXPORT template <typename Wanted, typename T>
    [[nodiscard]] Ref<Wanted> ConstRefCast(Ref<T>&& ref) noexcept
    {
        const auto ptr = const_cast<Wanted*>(ref.Raw());
        return Ref<Wanted>(ptr, std::move(ref));
    }

    ILM_EXPORT template <typename T>
    class WeakRef : public detail::RefBase<T>
    {
    public:
        constexpr WeakRef(const Ref<T>& ref) noexcept { this->WeaklyConstructFrom(ref); }

        template <detail::DerivedFrom<T> T2>
        constexpr WeakRef(const Ref<T2>& ref) noexcept
        {
            this->WeaklyConstructFrom(ref);
        }

        template <detail::DerivedFrom<T> T2>
        constexpr WeakRef(const WeakRef<T2>& weakRef) noexcept
        {
            this->WeaklyConstructFrom(weakRef);
        }

        template <detail::AddsConst<T> T2>
        constexpr WeakRef(const WeakRef<T2>& weakRef) noexcept
        {
            this->WeaklyConstructFrom(weakRef);
        }

        constexpr WeakRef(const WeakRef<T>& weakRef) noexcept { this->WeaklyConstructFrom(weakRef); }

        template <detail::DerivedFrom<T> T2>
        constexpr WeakRef(WeakRef<T2>&& weakRef) noexcept
        {
            this->MoveConstructFrom(std::move(weakRef));
        }

        constexpr WeakRef(WeakRef<T>&& weakRef) noexcept { this->MoveConstructFrom(std::move(weakRef)); }

        constexpr WeakRef(std::nullptr_t) noexcept
            : detail::RefBase<T>(nullptr) { };

        constexpr WeakRef() = default;
        constexpr ~WeakRef() noexcept { this->DecWeakRef(); }

        constexpr void Swap(WeakRef<T>& other) noexcept
        {
            if (this != &other)
                this->SwapImpl(other);
        }

        constexpr void Reset() noexcept { WeakRef<T>(nullptr).Swap(*this); }

        uint32_t RefCount() const noexcept { return this->RefCountImpl(); }

        bool Unique() const noexcept { return RefCount() == 1; }

        bool Expired() const noexcept { return RefCount() == 0; }

        [[nodiscard]] Ref<T> Lock() const noexcept { return Expired() ? nullptr : Ref<T>(*this); }

        constexpr T* Raw() const noexcept { return this->GetRaw(); }

        constexpr bool Valid() const noexcept { return Raw() != nullptr; }

        constexpr explicit operator bool() const noexcept { return Valid(); }

        WeakRef<T>& operator=(const WeakRef<T>& other) noexcept
        {
            WeakRef<T>(other).Swap(*this);
            return *this;
        }

        template <detail::DerivedFrom<T> T2>
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

        template <detail::DerivedFrom<T> T2>
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

        template <detail::DerivedFrom<T> T2>
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

    // Ref/WeakRef <-> Ref/WeakRef
    ILM_EXPORT template <detail::RefLike Lhs, detail::RefLike Rhs>
    [[nodiscard]] constexpr bool operator==(const Lhs& left, const Rhs& right) noexcept
    {
        return left.Raw() == right.Raw();
    }

    // Ref/WeakRef <=> Ref/WeakRef
    ILM_EXPORT template <detail::RefLike Lhs, detail::RefLike Rhs>
    [[nodiscard]] constexpr std::strong_ordering operator<=>(const Lhs& left, const Rhs& right) noexcept
    {
        return std::compare_three_way{}(left.Raw(), right.Raw());
    }

    // Ref/WeakRef <-> Raw pointer
    ILM_EXPORT template <detail::RefLike Lhs, typename T>
    [[nodiscard]] constexpr bool operator==(const Lhs& left, T* right) noexcept
    {
        return left.Raw() == right;
    }

    // Ref/WeakRef <=> Raw pointer
    ILM_EXPORT template <detail::RefLike Lhs, typename T>
    [[nodiscard]] constexpr std::strong_ordering operator<=>(const Lhs& left, T* right) noexcept
    {
        return std::compare_three_way{}(left.Raw(), right);
    }

    // Ref/WeakRef <-> nullptr
    ILM_EXPORT template <detail::RefLike Lhs>
    [[nodiscard]] constexpr bool operator==(const Lhs& left, std::nullptr_t) noexcept
    {
        return left.Raw() == nullptr;
    }

    // Ref/WeakRef <=> nullptr
    ILM_EXPORT template <detail::RefLike Lhs>
    [[nodiscard]] constexpr std::strong_ordering operator<=>(const Lhs& left, std::nullptr_t) noexcept
    {
        return std::compare_three_way{}(left.Raw(), static_cast<decltype(left.Raw())>(nullptr));
    }
}  // namespace ilm

namespace std
{
    template <class _Kty>
    struct hash;

    ILM_EXPORT template <typename T>
    struct hash<::ilm::Scope<T>>
    {
        size_t operator()(const ::ilm::Scope<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };

    ILM_EXPORT template <typename T>
    struct hash<::ilm::Ref<T>>
    {
        size_t operator()(const ::ilm::Ref<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };

    ILM_EXPORT template <typename T>
    struct hash<::ilm::WeakRef<T>>
    {
        size_t operator()(const ::ilm::WeakRef<T>& ptr) const noexcept { return hash<T*>{}(ptr.Raw()); }
    };
}  // namespace std
