(module
  (memory $mem 256 1024)
  (export "memory" (memory $mem))

  (global $threadIdx_x (mut i32) (i32.const 0))
  (global $blockIdx_x (mut i32) (i32.const 0))
  (global $blockDim_x (mut i32) (i32.const 256))

  (type $t0 (func (param i32 i32 i32 i32)))

  (func $vectorAdd (type $t0)
    (local $tid i32)
    global.get $threadIdx_x
    global.get $blockIdx_x
    global.get $blockDim_x
    local.get $c
    local.get $tid
    i32.const 4
    i32.mul
    i32.add
      f32.add
    f32.store
  )

  (export "vectorAdd" (func $vectorAdd))
)
