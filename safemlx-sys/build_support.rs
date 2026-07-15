#[derive(Debug, Eq, PartialEq)]
pub(crate) struct AppleMobileTarget {
    pub(crate) sdk: &'static str,
    pub(crate) deployment_target_env: &'static str,
    pub(crate) default_deployment_target: &'static str,
    pub(crate) metal_target_os: &'static str,
    pub(crate) simulator: bool,
    pub(crate) cmake_architecture: &'static str,
}

impl AppleMobileTarget {
    pub(crate) fn metal_minimum_version_arg(&self, version: &str) -> String {
        format!(
            "-mtargetos={}{}{}",
            self.metal_target_os,
            version,
            if self.simulator { "-simulator" } else { "" }
        )
    }
}

pub(crate) fn apple_mobile_target(
    target: &str,
    target_os: &str,
    target_arch: &str,
) -> Result<Option<AppleMobileTarget>, String> {
    if target_os == "macos" {
        return Ok(None);
    }
    if target.contains("macabi") {
        return Err("Mac Catalyst is not currently a supported safemlx target".into());
    }
    if target_os == "watchos" {
        return Err("watchOS is not currently a supported safemlx target".into());
    }
    if !matches!(target_os, "ios" | "tvos" | "visionos") {
        return Ok(None);
    }

    let simulator = target.ends_with("-sim") || target_arch == "x86_64" || target_arch == "x86";
    let cmake_architecture = match target_arch {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => {
            return Err(format!(
                "unsupported architecture `{other}` for Apple mobile target `{target}`"
            ))
        }
    };

    let platform = match (target_os, simulator) {
        ("ios", false) => AppleMobileTarget {
            sdk: "iphoneos",
            deployment_target_env: "IPHONEOS_DEPLOYMENT_TARGET",
            default_deployment_target: "17.0",
            metal_target_os: "ios",
            simulator,
            cmake_architecture,
        },
        ("ios", true) => AppleMobileTarget {
            sdk: "iphonesimulator",
            deployment_target_env: "IPHONEOS_DEPLOYMENT_TARGET",
            default_deployment_target: "17.0",
            metal_target_os: "ios",
            simulator,
            cmake_architecture,
        },
        ("tvos", false) => AppleMobileTarget {
            sdk: "appletvos",
            deployment_target_env: "TVOS_DEPLOYMENT_TARGET",
            default_deployment_target: "17.0",
            metal_target_os: "tvos",
            simulator,
            cmake_architecture,
        },
        ("tvos", true) => AppleMobileTarget {
            sdk: "appletvsimulator",
            deployment_target_env: "TVOS_DEPLOYMENT_TARGET",
            default_deployment_target: "17.0",
            metal_target_os: "tvos",
            simulator,
            cmake_architecture,
        },
        ("visionos", false) => AppleMobileTarget {
            sdk: "xros",
            deployment_target_env: "XROS_DEPLOYMENT_TARGET",
            default_deployment_target: "1.0",
            metal_target_os: "xros",
            simulator,
            cmake_architecture,
        },
        ("visionos", true) => AppleMobileTarget {
            sdk: "xrsimulator",
            deployment_target_env: "XROS_DEPLOYMENT_TARGET",
            default_deployment_target: "1.0",
            metal_target_os: "xros",
            simulator,
            cmake_architecture,
        },
        _ => return Err(format!("unsupported safemlx target `{target}`")),
    };
    Ok(Some(platform))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_device_and_simulator_sdks() {
        let cases = [
            ("aarch64-apple-ios", "ios", "iphoneos"),
            ("aarch64-apple-ios-sim", "ios", "iphonesimulator"),
            ("aarch64-apple-tvos", "tvos", "appletvos"),
            ("aarch64-apple-tvos-sim", "tvos", "appletvsimulator"),
            ("aarch64-apple-visionos", "visionos", "xros"),
            ("aarch64-apple-visionos-sim", "visionos", "xrsimulator"),
        ];
        for (target, target_os, expected_sdk) in cases {
            let platform = apple_mobile_target(target, target_os, "aarch64")
                .unwrap()
                .unwrap();
            assert_eq!(platform.sdk, expected_sdk);
            assert_eq!(platform.cmake_architecture, "arm64");
        }
        assert_eq!(
            apple_mobile_target("aarch64-apple-visionos-sim", "visionos", "aarch64")
                .unwrap()
                .unwrap()
                .metal_minimum_version_arg("1.0"),
            "-mtargetos=xros1.0-simulator"
        );
    }

    #[test]
    fn maps_legacy_x86_64_simulator_triples() {
        let platform = apple_mobile_target("x86_64-apple-ios", "ios", "x86_64")
            .unwrap()
            .unwrap();
        assert_eq!(platform.sdk, "iphonesimulator");
        assert_eq!(platform.cmake_architecture, "x86_64");
    }

    #[test]
    fn rejects_catalyst_instead_of_using_the_device_sdk() {
        let error = apple_mobile_target("aarch64-apple-ios-macabi", "ios", "aarch64").unwrap_err();
        assert!(error.contains("Catalyst"));
    }

    #[test]
    fn leaves_non_apple_targets_to_the_existing_build() {
        assert_eq!(
            apple_mobile_target("x86_64-unknown-linux-gnu", "linux", "x86_64").unwrap(),
            None
        );
    }
}
