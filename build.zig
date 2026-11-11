const std = @import("std");

fn configure(mod: *std.Build.Module, b: *std.Build, deps_prefix: ?[]const u8, cuda_prefix: ?[]const u8, cuda: bool, link_deps: bool) void {
    mod.addIncludePath(b.path("include"));
    if (deps_prefix) |prefix| {
        mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) });
    }
    if (cuda_prefix) |prefix| {
        mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib64" }) });
    }
    if (cuda) mod.addCMacro("UMMINFER_WITH_CUDA", "1");
    if (link_deps) {
        mod.linkSystemLibrary("zmq", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("msgpackc", .{ .use_pkg_config = .no });
        if (cuda) mod.linkSystemLibrary("cudart", .{ .use_pkg_config = .no });
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
    const cuda_prefix = b.option([]const u8, "cuda-prefix", "CUDA prefix containing include/ and lib64/");
    const cuda = b.option(bool, "cuda", "Enable the CUDA IPC relay") orelse (cuda_prefix != null);

    const core = b.addLibrary(.{
        .name = "umm_core",
        .linkage = .static,
        .root_module = module(b, target, optimize),
    });
    configure(core.root_module, b, deps_prefix, cuda_prefix, cuda, false);
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
        .name = "umm-infer",
        .root_module = module(b, target, optimize),
    });
    configure(exe.root_module, b, deps_prefix, cuda_prefix, cuda, true);
    exe.root_module.linkLibrary(core);
    exe.root_module.addCSourceFiles(.{
        .files = &.{"src/main.cpp"},
        .flags = &.{"-std=c++20"},
    });
    b.installArtifact(exe);

    const test_step = b.step("test", "Run C++ runtime tests");
    var ipc_exe: ?*std.Build.Step.Compile = null;
    var pipeline_exe: ?*std.Build.Step.Compile = null;
    const tests = [_]struct { name: []const u8, source: []const u8, cuda: bool = false }{
        .{ .name = "self_check", .source = "tests/self_check.cpp" },
        .{ .name = "wire_compat", .source = "tests/compat/wire_compat.cpp" },
        .{ .name = "runtime_compat", .source = "tests/compat/runtime_compat.cpp" },
        .{ .name = "cuda_ipc_test", .source = "tests/cuda_ipc_test.cpp", .cuda = true },
        .{ .name = "cuda_pipeline_test", .source = "tests/cuda_pipeline_test.cpp", .cuda = true },
    };
    for (tests) |item| {
        if (item.cuda and !cuda) continue;
        const test_exe = b.addExecutable(.{
            .name = item.name,
            .root_module = module(b, target, optimize),
        });
        configure(test_exe.root_module, b, deps_prefix, cuda_prefix, cuda, true);
        test_exe.root_module.linkLibrary(core);
        test_exe.root_module.addCSourceFiles(.{
            .files = &.{item.source},
            .flags = &.{"-std=c++20"},
        });
        if (item.cuda) {
            b.installArtifact(test_exe);
            if (std.mem.eql(u8, item.name, "cuda_ipc_test")) ipc_exe = test_exe;
            if (std.mem.eql(u8, item.name, "cuda_pipeline_test")) pipeline_exe = test_exe;
        }
        const run = b.addRunArtifact(test_exe);
        test_step.dependOn(&run.step);
    }

    const stress_step = b.step("cuda-stress", "Repeat the CUDA tests until a failure");
    if (ipc_exe != null and pipeline_exe != null) {
        const stress = b.addSystemCommand(&.{ "python3", "-I" });
        stress.addFileArg(b.path("tests/cuda_stress.py"));
        stress.addArgs(&.{"--ipc"});
        stress.addArtifactArg(ipc_exe.?);
        stress.addArgs(&.{"--pipeline"});
        stress.addArtifactArg(pipeline_exe.?);
        stress_step.dependOn(&stress.step);
    } else {
        const note = b.addSystemCommand(&.{ "sh", "-c", "echo 'cuda-stress needs -Dcuda; no CUDA test binaries were built'" });
        stress_step.dependOn(&note.step);
    }

    const bench_step = b.step("cuda-bench", "Measure SHM and CUDA IPC relay latency");
    if (cuda) {
        const bench_exe = b.addExecutable(.{
            .name = "cuda_bench",
            .root_module = module(b, target, optimize),
        });
        configure(bench_exe.root_module, b, deps_prefix, cuda_prefix, cuda, true);
        bench_exe.root_module.linkLibrary(core);
        bench_exe.root_module.addCSourceFiles(.{
            .files = &.{"tests/cuda_bench.cpp"},
            .flags = &.{"-std=c++20"},
        });
        b.installArtifact(bench_exe);
        const run = b.addRunArtifact(bench_exe);
        bench_step.dependOn(&run.step);
    } else {
        const note = b.addSystemCommand(&.{ "sh", "-c", "echo 'cuda-bench needs -Dcuda; no benchmark built'" });
        bench_step.dependOn(&note.step);
    }

    const smoke = b.addSystemCommand(&.{ "python3", "-I" });
    smoke.addFileArg(b.path("tests/http_smoke.py"));
    smoke.addArtifactArg(exe);
    const smoke_step = b.step("smoke", "Run HTTP and process lifecycle tests");
    smoke_step.dependOn(&smoke.step);
}
