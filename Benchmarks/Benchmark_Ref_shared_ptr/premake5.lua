project "Benchmark_Ref_shared_ptr"
    kind "ConsoleApp"
    language "C++"

    debugdir(OUT_DIR)
    targetdir(OUT_DIR)
    objdir(INT_DIR)

    files
    {
        "./*.hpp",
        "./*.cpp"
    }

    includedirs
    {
        ".",
        "%{ILM_HPP_INCLUDE}"
    }

    filter "system:linux"
        links { "pthread" }
