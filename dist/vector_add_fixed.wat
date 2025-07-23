(module
  (memory $mem 256 1024)
  (export "memory" (memory $mem))

  (global $threadIdx_x (mut i32) (i32.const 0))
  (global $blockIdx_x (mut i32) (i32.const 0))
  (global $blockDim_x (mut i32) (i32.const 256))

  (type $t0 (func (param $a i32) (param $b i32) (param $c i32) (param $n i32)))

  (func $vectorAdd (type $t0) (param $a i32) (param $b i32) (param $c i32) (param $n i32)
    (local $tid i32)
    (local $i i32)
    
    ;; Calculate thread ID: blockIdx.x * blockDim.x + threadIdx.x
    global.get $blockIdx_x
    global.get $blockDim_x
    i32.mul
    global.get $threadIdx_x
    i32.add
    local.set $tid
    
    ;; Loop through elements (simplified for WebAssembly)
    (loop $loop
      ;; Check bounds
      local.get $i
      local.get $n
      i32.ge_s
      br_if 1
      
      ;; c[i] = a[i] + b[i]
      local.get $c
      local.get $i
      i32.const 4
      i32.mul
      i32.add
      
      local.get $a
      local.get $i
      i32.const 4
      i32.mul
      i32.add
      f32.load
      
      local.get $b
      local.get $i
      i32.const 4
      i32.mul
      i32.add
      f32.load
      
      f32.add
      f32.store
      
      ;; Increment counter
      local.get $i
      i32.const 1
      i32.add
      local.set $i
      
      br $loop
    )
  )

  (func $matrixMul (type $t0) (param $a i32) (param $b i32) (param $c i32) (param $size i32)
    (local $i i32)
    (local $j i32)
    (local $k i32)
    (local $sum f32)
    
    ;; Simplified matrix multiplication
    (loop $outer
      local.get $i
      local.get $size
      i32.ge_s
      br_if 1
      
      i32.const 0
      local.set $j
      
      (loop $inner
        local.get $j
        local.get $size
        i32.ge_s
        br_if 1
        
        f32.const 0
        local.set $sum
        
        i32.const 0
        local.set $k
        
        (loop $dot
          local.get $k
          local.get $size
          i32.ge_s
          br_if 1
          
          ;; sum += a[i*size + k] * b[k*size + j]
          local.get $sum
          
          local.get $a
          local.get $i
          local.get $size
          i32.mul
          local.get $k
          i32.add
          i32.const 4
          i32.mul
          i32.add
          f32.load
          
          local.get $b
          local.get $k
          local.get $size
          i32.mul
          local.get $j
          i32.add
          i32.const 4
          i32.mul
          i32.add
          f32.load
          
          f32.mul
          f32.add
          local.set $sum
          
          local.get $k
          i32.const 1
          i32.add
          local.set $k
          
          br $dot
        )
        
        ;; c[i*size + j] = sum
        local.get $c
        local.get $i
        local.get $size
        i32.mul
        local.get $j
        i32.add
        i32.const 4
        i32.mul
        i32.add
        local.get $sum
        f32.store
        
        local.get $j
        i32.const 1
        i32.add
        local.set $j
        
        br $inner
      )
      
      local.get $i
      i32.const 1
      i32.add
      local.set $i
      
      br $outer
    )
  )

  (export "vectorAdd" (func $vectorAdd))
  (export "matrixMul" (func $matrixMul))
)