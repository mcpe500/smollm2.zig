const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "smollm2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .link_libc = true,
        }),
    });

    exe.linkSystemLibrary("m");

    b.installArtifact(exe);
}
