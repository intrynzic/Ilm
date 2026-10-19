include "Tools/Premake/customization/solutionitems.lua"

local function PickCppVersion(targetVersion)
    if os.target() == "windows" then return "C++" .. targetVersion end
    if os.target() == "linux" then return "gnu++" .. targetVersion end

    return "Default"
end

-- C++26
CPP_VER = PickCppVersion("latest")
print("C++ Standard: " .. CPP_VER .. ".\n")

OUT_DIR = "%{wks.location}/bin/build/%{cfg.system}/%{cfg.architecture}/%{cfg.buildcfg}"
INT_DIR = "%{wks.location}/bin/intermediate/%{cfg.system}/%{cfg.architecture}/%{cfg.buildcfg}/%{prj.name}"

-- This path is relative to the premake scripts for each Example/Test, not relative to this premake script
ILM_HPP_INCLUDE = "../../Ilm/src/include"

include "Examples"
include "Tests"
