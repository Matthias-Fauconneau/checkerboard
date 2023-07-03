use image::Image;
pub fn upscale(target: &mut Image<&mut [u32]>, source: Image<&[u8]>) {
    let [num, den] = if source.size.x*target.size.y > source.size.y*target.size.x { [target.size.x, source.size.x] } else { [target.size.y, source.size.y] };
    let target_size = source.size*(num/den); // largest integer fit
    let mut target = target.slice_mut((target.size-target_size)/2, target_size);
    let factor = target.size.x/source.size.x;
    let stride_factor = target.stride*factor;
    let mut row = target.as_mut_ptr();
    for y in 0..source.size.y {
        {
            let mut row = row;
            for x in 0..source.size.x {
                let value = source[xy{x,y}];
                let p = value as u32 | (value as u32)<<8 | (value as u32)<<16;
                let p4 = [p; 4];
                {
                    let mut row = row;
                    for _ in 0..factor { unsafe{
                        {
                            let mut row = row;
                            for _ in 0..factor/4 {
                                (row as *mut [u32; 4]).write_unaligned(p4);
                                row = row.add(4);
                            }
                            for _ in factor/4*4..factor {
                                *row = p;
                                row = row.add(1);
                            }
                        }
                        row = row.add(target.stride as usize);
                    }}
                }
                row = unsafe{row.add(factor as usize)};
            }
        }
        row = unsafe{row.add(stride_factor as usize)};
    }
}

use {vector::{xy, uint2, vec2}, crate::matrix::{mat3, apply}};
pub fn affine_blit(target: &mut Image<&mut[u8]>, source: Image<&[u8]>, A: mat3) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let p = vec2::from(xy{x,y}.signed());
        //let p = vec2::from(source.size.signed())/vec2::from(target.size.signed())*p;
        let p = apply(A, p);
        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
        target[xy{x, y}] = source[uint2::from(p)];
    }}
}
