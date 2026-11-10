const std = @import("std");

fn configure(mod: *std.Build.Module, b: *std.Build, deps_prefix: ?[]const u8, link_deps: bool) void {
    mod.addIncludePath(b.path("include"));
    if (deps_prefix) |prefix| {
        mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) });
    }
    if (link_deps) {
        mod.linkSystemLibrary("zmq", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("msgpackc", .{ .use_pkg_config = .no });
    }
}

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const deps_prefix = b.option([]const u8, "deps-prefix", "Prefix containing include/ and lib/ for native dependencies");

    const core = b.addLibrary(.{
        .name = "omni_core",
        .linkage = .static,
        .root_module = module(b, target, optimize),
    });
    configure(core.root_module, b, deps_prefix, false);
    core.root_module.addCSourceFiles(.{
        .files = &.{
            "src/proto/wire.cpp",
            "src/proto/messages.cpp",
            "src/relay/relay.cpp",
            "src/comm/comm.cpp",
            "src/scheduling/simple_scheduler.cpp",
            "src/pipeline/pipeline.cpp",
            "src/serve/openai_api.cpp",
        },
        .flags = &.{"-std=c++20"},
    });

    const exe = b.addExecutable(.{
        .name = "omni-infer",
        .root_module = module(b, target, optimize),
    });
    configure(exe.root_module, b, deps_prefix, true);
    exe.root_module.linkLibrary(core);
    exe.root_module.addCSourceFiles(.{
        .files = &.{"src/main.cpp"},
        .flags = &.{"-std=c++20"},
    });
    b.installArtifact(exe);

    const test_step = b.step("test", "Run C++ runtime tests");
    const tests = [_]struct { name: []const u8, source: []const u8 }{
        .{ .name = "self_check", .source = "tests/self_check.cpp" },
        .{ .name = "wire_compat", .source = "tests/compat/wire_compat.cpp" },
        .{ .name = "runtime_compat", .source = "tests/compat/runtime_compat.cpp" },
    };
    for (tests) |item| {
        const test_exe = b.addExecutable(.{
            .name = item.name,
            .root_module = module(b, target, optimize),
        });
        configure(test_exe.root_module, b, deps_prefix, true);
        test_exe.root_module.linkLibrary(core);
        test_exe.root_module.addCSourceFiles(.{
            .files = &.{item.source},
            .flags = &.{"-std=c++20"},
        });
        const run = b.addRunArtifact(test_exe);
        test_step.dependOn(&run.step);
    }

    const smoke = b.addSystemCommand(&.{ "python3", "-I" });
    smoke.addFileArg(b.path("tests/http_smoke.py"));
    smoke.addArtifactArg(exe);
    const smoke_step = b.step("smoke", "Run HTTP and process lifecycle tests");
    smoke_step.dependOn(&smoke.step);
}
