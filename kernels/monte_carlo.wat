(module
  ;; Import memory from environment
  (import "env" "memory" (memory 256))
  
  ;; Export memory
  (export "memory" (memory 0))
  
  ;; Simple linear congruential generator for pseudo-random numbers
  (func $random (param $seed i32) (result f32)
    (local $a i32)
    (local $c i32) 
    (local $result i32)
    
    ;; LCG parameters: a = 1664525, c = 1013904223
    (local.set $a (i32.const 1664525))
    (local.set $c (i32.const 1013904223))
    
    ;; result = (a * seed + c) & 0x7FFFFFFF
    (local.set $result 
      (i32.and 
        (i32.add 
          (i32.mul (local.get $seed) (local.get $a))
          (local.get $c)
        )
        (i32.const 0x7FFFFFFF)
      )
    )
    
    ;; Convert to float [0, 1)
    (f32.div 
      (f32.convert_i32_u (local.get $result))
      (f32.const 2147483648.0) ;; 2^31
    )
  )
  
  ;; Monte Carlo Pi estimation
  (func $monteCarloPi (param $samples i32) (result f32)
    (local $i i32)
    (local $inside i32)
    (local $x f32)
    (local $y f32)
    (local $distance f32)
    (local $seed1 i32)
    (local $seed2 i32)
    
    ;; Initialize seeds
    (local.set $seed1 (i32.const 12345))
    (local.set $seed2 (i32.const 67890))
    (local.set $inside (i32.const 0))
    (local.set $i (i32.const 0))
    
    ;; Main loop
    (loop $main_loop
      ;; Generate x coordinate
      (local.set $seed1 
        (i32.and 
          (i32.add 
            (i32.mul (local.get $seed1) (i32.const 1664525))
            (i32.const 1013904223)
          )
          (i32.const 0x7FFFFFFF)
        )
      )
      (local.set $x (call $random (local.get $seed1)))
      
      ;; Generate y coordinate  
      (local.set $seed2
        (i32.and 
          (i32.add 
            (i32.mul (local.get $seed2) (i32.const 1103515245))
            (i32.const 12345)
          )
          (i32.const 0x7FFFFFFF)
        )
      )
      (local.set $y (call $random (local.get $seed2)))
      
      ;; Calculate distance^2 = x^2 + y^2
      (local.set $distance 
        (f32.add 
          (f32.mul (local.get $x) (local.get $x))
          (f32.mul (local.get $y) (local.get $y))
        )
      )
      
      ;; Check if inside unit circle
      (if (f32.le (local.get $distance) (f32.const 1.0))
        (then
          (local.set $inside (i32.add (local.get $inside) (i32.const 1)))
        )
      )
      
      ;; Increment counter
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      
      ;; Continue if not done
      (br_if $main_loop (i32.lt_s (local.get $i) (local.get $samples)))
    )
    
    ;; Return 4 * (inside / samples)
    (f32.mul 
      (f32.const 4.0)
      (f32.div 
        (f32.convert_i32_s (local.get $inside))
        (f32.convert_i32_s (local.get $samples))
      )
    )
  )
  
  ;; Vector addition function (existing)
  (func $vectorAdd (param $ptrA i32) (param $ptrB i32) (param $ptrC i32) (param $length i32)
    (local $i i32)
    (local $a f32)
    (local $b f32)
    (local $c f32)
    
    (local.set $i (i32.const 0))
    
    (loop $add_loop
      ;; Load A[i] and B[i]
      (local.set $a (f32.load (i32.add (local.get $ptrA) (i32.mul (local.get $i) (i32.const 4)))))
      (local.set $b (f32.load (i32.add (local.get $ptrB) (i32.mul (local.get $i) (i32.const 4)))))
      
      ;; C[i] = A[i] + B[i]
      (local.set $c (f32.add (local.get $a) (local.get $b)))
      (f32.store (i32.add (local.get $ptrC) (i32.mul (local.get $i) (i32.const 4))) (local.get $c))
      
      ;; i++
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      
      ;; Continue if i < length
      (br_if $add_loop (i32.lt_s (local.get $i) (local.get $length)))
    )
  )
  
  ;; Parallel sum function
  (func $parallelSum (param $dataPtr i32) (param $length i32) (result f32)
    (local $i i32)
    (local $sum f32)
    
    (local.set $sum (f32.const 0.0))
    (local.set $i (i32.const 0))
    
    (loop $sum_loop
      (local.set $sum
        (f32.add
          (local.get $sum)
          (f32.load (i32.add (local.get $dataPtr) (i32.mul (local.get $i) (i32.const 4))))
        )
      )
      
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $sum_loop (i32.lt_s (local.get $i) (local.get $length)))
    )
    
    (local.get $sum)
  )
  
  ;; Gaussian blur function (3x3 kernel)
  (func $gaussianBlur (param $inputPtr i32) (param $outputPtr i32) (param $width i32) (param $height i32)
    (local $x i32)
    (local $y i32)
    (local $r f32)
    (local $g f32)
    (local $b f32)
    (local $a f32)
    (local $kx i32)
    (local $ky i32)
    (local $nx i32)
    (local $ny i32)
    (local $pixelIdx i32)
    (local $pixel i32)
    (local $weight f32)
    
    ;; Gaussian kernel weights (normalized 3x3)
    ;; 0.0625 0.125 0.0625
    ;; 0.125  0.25  0.125
    ;; 0.0625 0.125 0.0625
    
    (local.set $y (i32.const 0))
    (loop $y_loop
      (local.set $x (i32.const 0))
      (loop $x_loop
        ;; Initialize accumulation
        (local.set $r (f32.const 0.0))
        (local.set $g (f32.const 0.0))
        (local.set $b (f32.const 0.0))
        (local.set $a (f32.const 0.0))
        
        ;; Apply 3x3 kernel
        (local.set $ky (i32.const -1))
        (loop $ky_loop
          (local.set $kx (i32.const -1))
          (loop $kx_loop
            ;; Calculate neighbor coordinates (clamped)
            (local.set $nx
              (i32.add (local.get $x) (local.get $kx))
            )
            (if (i32.lt_s (local.get $nx) (i32.const 0))
              (then (local.set $nx (i32.const 0)))
            )
            (if (i32.ge_s (local.get $nx) (local.get $width))
              (then (local.set $nx (i32.sub (local.get $width) (i32.const 1))))
            )
            
            (local.set $ny
              (i32.add (local.get $y) (local.get $ky))
            )
            (if (i32.lt_s (local.get $ny) (i32.const 0))
              (then (local.set $ny (i32.const 0)))
            )
            (if (i32.ge_s (local.get $ny) (local.get $height))
              (then (local.set $ny (i32.sub (local.get $height) (i32.const 1))))
            )
            
            ;; Get kernel weight
            (local.set $weight
              (if (result f32)
                (i32.and (i32.eq (local.get $kx) (i32.const 0)) (i32.eq (local.get $ky) (i32.const 0)))
                (then (f32.const 0.25))  ;; center
                (else
                  (if (result f32)
                    (i32.or (i32.eq (local.get $kx) (i32.const 0)) (i32.eq (local.get $ky) (i32.const 0)))
                    (then (f32.const 0.125))  ;; edge
                    (else (f32.const 0.0625)) ;; corner
                  )
                )
              )
            )
            
            ;; Load pixel (simplified as grayscale for now)
            (local.set $pixelIdx
              (i32.add
                (local.get $inputPtr)
                (i32.mul (i32.add (i32.mul (local.get $ny) (local.get $width)) (local.get $nx)) (i32.const 4))
              )
            )
            
            ;; Accumulate weighted values
            (local.set $r
              (f32.add
                (local.get $r)
                (f32.mul (local.get $weight) (f32.convert_i32_u (i32.load8_u (local.get $pixelIdx))))
              )
            )
            
            (local.set $kx (i32.add (local.get $kx) (i32.const 1)))
            (br_if $kx_loop (i32.le_s (local.get $kx) (i32.const 1)))
          )
          (local.set $ky (i32.add (local.get $ky) (i32.const 1)))
          (br_if $ky_loop (i32.le_s (local.get $ky) (i32.const 1)))
        )
        
        ;; Store result
        (local.set $pixelIdx
          (i32.add
            (local.get $outputPtr)
            (i32.mul (i32.add (i32.mul (local.get $y) (local.get $width)) (local.get $x)) (i32.const 4))
          )
        )
        (i32.store8 (local.get $pixelIdx) (i32.trunc_f32_u (local.get $r)))
        
        (local.set $x (i32.add (local.get $x) (i32.const 1)))
        (br_if $x_loop (i32.lt_s (local.get $x) (local.get $width)))
      )
      (local.set $y (i32.add (local.get $y) (i32.const 1)))
      (br_if $y_loop (i32.lt_s (local.get $y) (local.get $height)))
    )
  )
  
  ;; Simple 1D FFT (power of 2 sizes only)
  (func $fft (param $realPtr i32) (param $imagPtr i32) (param $N i32)
    (local $i i32)
    (local $j i32)
    (local $k i32)
    (local $m i32)
    (local $temp_r f32)
    (local $temp_i f32)
    (local $u_r f32)
    (local $u_i f32)
    (local $w_r f32)
    (local $w_i f32)
    (local $angle f32)
    
    ;; Bit-reversal permutation
    (local.set $j (i32.const 0))
    (local.set $i (i32.const 1))
    (loop $bit_reverse_outer
      (if (i32.lt_s (local.get $i) (local.get $j))
        (then
          ;; Swap real parts
          (local.set $temp_r (f32.load (i32.add (local.get $realPtr) (i32.mul (local.get $i) (i32.const 4)))))
          (f32.store
            (i32.add (local.get $realPtr) (i32.mul (local.get $i) (i32.const 4)))
            (f32.load (i32.add (local.get $realPtr) (i32.mul (local.get $j) (i32.const 4))))
          )
          (f32.store
            (i32.add (local.get $realPtr) (i32.mul (local.get $j) (i32.const 4)))
            (local.get $temp_r)
          )
          
          ;; Swap imaginary parts
          (local.set $temp_i (f32.load (i32.add (local.get $imagPtr) (i32.mul (local.get $i) (i32.const 4)))))
          (f32.store
            (i32.add (local.get $imagPtr) (i32.mul (local.get $i) (i32.const 4)))
            (f32.load (i32.add (local.get $imagPtr) (i32.mul (local.get $j) (i32.const 4))))
          )
          (f32.store
            (i32.add (local.get $imagPtr) (i32.mul (local.get $j) (i32.const 4)))
            (local.get $temp_i)
          )
        )
      )
      
      (local.set $k (i32.shr_s (local.get $N) (i32.const 1)))
      (loop $bit_reverse_inner
        (if (i32.ge_s (local.get $j) (local.get $k))
          (then
            (local.set $j (i32.sub (local.get $j) (local.get $k)))
            (local.set $k (i32.shr_s (local.get $k) (i32.const 1)))
            (br_if $bit_reverse_inner (i32.gt_s (local.get $k) (i32.const 0)))
          )
          (else
            (br $bit_reverse_inner)
          )
        )
      )
      (local.set $j (i32.add (local.get $j) (local.get $k)))
      
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $bit_reverse_outer (i32.lt_s (local.get $i) (local.get $N)))
    )
    
    ;; Main FFT computation (simplified)
    (local.set $m (i32.const 2))
    (loop $fft_stages
      (local.set $i (i32.const 0))
      (loop $fft_groups
        (local.set $j (i32.const 0))
        (loop $fft_elements
          ;; Simple butterfly operation (without proper twiddle factors for brevity)
          (local.set $k (i32.add (local.get $j) (i32.shr_s (local.get $m) (i32.const 1))))
          
          (local.set $u_r (f32.load (i32.add (local.get $realPtr) (i32.mul (local.get $j) (i32.const 4)))))
          (local.set $u_i (f32.load (i32.add (local.get $imagPtr) (i32.mul (local.get $j) (i32.const 4)))))
          (local.set $temp_r (f32.load (i32.add (local.get $realPtr) (i32.mul (local.get $k) (i32.const 4)))))
          (local.set $temp_i (f32.load (i32.add (local.get $imagPtr) (i32.mul (local.get $k) (i32.const 4)))))
          
          (f32.store
            (i32.add (local.get $realPtr) (i32.mul (local.get $j) (i32.const 4)))
            (f32.add (local.get $u_r) (local.get $temp_r))
          )
          (f32.store
            (i32.add (local.get $imagPtr) (i32.mul (local.get $j) (i32.const 4)))
            (f32.add (local.get $u_i) (local.get $temp_i))
          )
          (f32.store
            (i32.add (local.get $realPtr) (i32.mul (local.get $k) (i32.const 4)))
            (f32.sub (local.get $u_r) (local.get $temp_r))
          )
          (f32.store
            (i32.add (local.get $imagPtr) (i32.mul (local.get $k) (i32.const 4)))
            (f32.sub (local.get $u_i) (local.get $temp_i))
          )
          
          (local.set $j (i32.add (local.get $j) (local.get $m)))
          (br_if $fft_elements (i32.lt_s (local.get $j) (local.get $N)))
        )
        (local.set $i (i32.add (local.get $i) (local.get $m)))
        (br_if $fft_groups (i32.lt_s (local.get $i) (local.get $N)))
      )
      (local.set $m (i32.shl (local.get $m) (i32.const 1)))
      (br_if $fft_stages (i32.le_s (local.get $m) (local.get $N)))
    )
  )
  
  ;; Export functions
  (export "monteCarloPi" (func $monteCarloPi))
  (export "vectorAdd" (func $vectorAdd))
  (export "parallelSum" (func $parallelSum))
  (export "gaussianBlur" (func $gaussianBlur))
  (export "fft" (func $fft))
)