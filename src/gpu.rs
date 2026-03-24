use anyhow::{Context, Result};
use log::info;
use std::mem;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixDims {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

pub async fn execute_matmul(
    bytes_a: &[u8],
    bytes_b: &[u8],
    m: u32,
    k: u32,
    n: u32,
) -> Result<Vec<f32>> {
    info!("Phase 3: Connecting to GPU and transferring Hugging Face weights to VRAM...");
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .context("Failed to locate a GPU adapter")?;

    let (device, queue) = adapter
        .request_device(&Default::default())
        .await
        .context("Failed to create logical device")?;

    let dims = MatrixDims { m, k, n, _padding: 0 };
    let output_byte_len = (m * n * 4) as wgpu::BufferAddress;

    let dims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims_buffer"),
        size: mem::size_of::<MatrixDims>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer_a_embedding"),
        size: bytes_a.len() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer_b_hf_weights"),
        size: bytes_b.len() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer_out"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));
    queue.write_buffer(&buffer_a, 0, bytes_a);
    queue.write_buffer(&buffer_b, 0, bytes_b);

    info!("Phase 4: Dispatching WGSL MatMul + ReLU on Real Output Grid...");

    let wgsl_shader = r#"
        struct MatrixDims {
            m: u32,
            k: u32,
            n: u32,
            _padding: u32,
        };

        @group(0) @binding(0) var<uniform> dims: MatrixDims;
        @group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
        @group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
        @group(0) @binding(3) var<storage, read_write> matrix_out: array<f32>;

        @compute @workgroup_size(1, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let col = global_id.x;
            let row = global_id.y;

            if (row >= dims.m || col >= dims.n) {
                return;
            }

            var sum = 0.0;
            for (var i = 0u; i < dims.k; i = i + 1u) {
                let a_val = matrix_a[row * dims.k + i];
                let b_val = matrix_b[i * dims.n + col];
                sum = sum + (a_val * b_val);
            }

            // Apply ReLU
            let relu_result = max(0.0, sum);
            matrix_out[row * dims.n + col] = relu_result;
        }
    "#;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MatMul Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl_shader.into()),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MatMul Pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Bindings"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_out.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(n, m, 1);
    }

    info!("Phase 5: Retrieving Calculated Vector from GPU...");

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: output_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging_buffer, 0, output_byte_len);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = sender.send(v);
    });

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .context("Failed to poll GPU device")?;

    receiver.await.context("Failed to wait for GPU receiver")?.context("Failed to map staging buffer")?;

    let view = buffer_slice.get_mapped_range();
    let final_data: &[f32] = bytemuck::cast_slice(&view);
    let result = final_data.to_vec();

    drop(view);
    staging_buffer.unmap();

    Ok(result)
}