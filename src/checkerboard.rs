#![allow(dead_code, unused_imports)]
use {num::{sq, zero}, vector::{xy, uint2, int2, vec2, cross2, norm, minmax, MinMax}, image::Image, ui::time};
use {std::{ops::Range, array::from_fn, simd::{Simd as SIMD, SimdUint as _, SimdMutPtr as _, SimdOrd as _, u8x32, u16x32, u32x16, u32x8, u64x8}}, core::arch::x86_64::*};
unsafe fn _mm512_noop_low(a: __m512i) ->__m256i { _mm512_extracti64x4_epi64(a, 0) }
unsafe fn _mm512_high_to_low(a: __m512i) ->__m256i { _mm512_extracti64x4_epi64(a, 1) }
unsafe fn noop_low(a: impl Into<__m512i>) ->__m256i { _mm512_noop_low(a.into()) }
unsafe fn high_to_low(a: impl Into<__m512i>) ->__m256i { _mm512_high_to_low(a.into()) }
unsafe fn cast_u32(a: u16x32) -> [u32x16; 2] { [_mm512_cvtepu16_epi32(noop_low(a)).into(), _mm512_cvtepu16_epi32(high_to_low(a)).into()] }
fn mul(s: u32, v: [u32x16; 2]) -> [u32x16; 2] { [SIMD::splat(s)*v[0], SIMD::splat(s)*v[1]] }
fn add(a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] { [a[0]+b[0], a[1]+b[1]] }
fn sub(a: [u32x16; 2], b: [u32x16; 2]) -> [u32x16; 2] { [a[0]-b[0], a[1]-b[1]] }
unsafe fn scatter(slice: *mut u16, offsets: [u32x16; 2], src: [u32x16; 2]) { // Scatter 32bit words in 16bit cells. Only works for increasing offsets
    _mm512_i32scatter_epi32(slice as *mut _, offsets[0].into(), src[0].into(), 2);
    _mm512_i32scatter_epi32(slice as *mut _, offsets[1].into(), src[1].into(), 2);
}
fn srl<const N: usize>(a: SIMD<u32, N>, count: u32) -> SIMD<u32,N> where std::simd::LaneCount<N>: std::simd::SupportedLaneCount { a >> SIMD::splat(count) }
fn sll<const N: usize>(a: SIMD<u32, N>, count: u32) -> SIMD<u32,N> where std::simd::LaneCount<N>: std::simd::SupportedLaneCount { a << SIMD::splat(count) }

pub fn max(data: &[u16]) -> u16 {
    let Range{start, end} = data.as_ptr_range();
    let [mut sample, end] = [start, end].map(|p| p as *const u16x32);
    let mut max = SIMD::splat(0);
    unsafe{while sample < end { max = max.simd_max(sample.read()); sample = sample.add(1); }};
    max.reduce_max()
}

const fn headroom(R: u32) -> u32 { (R+1+R).next_power_of_two().ilog2() }
pub fn transpose_box_convolve_scale<const R: u32, const THREADS : usize>(source: Image<&[u16]>, factor: u32) -> Image<Box<[u16]>> {
    let mut transpose = Image::<Box<[u16]>>::uninitialized(source.size.yx());
    {
        assert!(R+1+R < (1<<headroom(R)) && R+1+R > (1<<(headroom(R)-1)), "{R} {}", headroom(R));
        assert!(source.size.x%32 == 0, "{}", source.size.x); // == transpose height
        let source_stride = source.stride as usize;
        assert!(transpose.stride == transpose.size.x);
        let transpose_stride = transpose.stride;
        assert!(transpose_stride == source.size.y);
        
        let ref task = |i0, i1, transpose_chunk:Image<&mut[u16]>| unsafe {
            let Range{start, end} = source.data.as_ptr_range();
            assert_eq!(end, start.add((source.size.y-1)as usize*source_stride).add(source.size.x as usize));
            let [column, column_end] = [i0,i1].map(|i| start.add(i as usize));
            let [mut column, column_end, end] = [column, column_end, end].map(|p| p as *const u16x32);
        
            const U16 : usize = std::mem::size_of::<u16>();
            let transpose_chunk = transpose_chunk.data.as_mut_ptr();
            let mut rows = from_fn(|i| u32x16::from_array(from_fn(|y| (i*16+y) as u32*transpose_stride)));
            while column < column_end /*for _x in x0/32..x1/32*/ {
                let first = cast_u32(column.read());
                let mut sum = mul(R as u32, first);
                let mut front = column;
                for _y in 0..R { sum = add(sum, cast_u32(front.read())); front = front.byte_add(source_stride*U16); }
                fn srl(a: [u32x16; 2], count: u32) -> [u32x16; 2] { [a[0] >> SIMD::splat(count), a[1] >> SIMD::splat(count)] }
                for _y in 0..R {
                    sum = add(sum, cast_u32(front.read()));
                    front = front.byte_add(source_stride*U16);
                    scatter(transpose_chunk, rows, srl(mul(factor, sum), 16-headroom(R))); // Scattering 32bit words to increasing offsets: high part only overwrites cells before assignment
                    sum = sub(sum, first);
                    rows = add(rows, [SIMD::splat(1); 2]);
                }
                let mut back = column;
                let mut last = first;
                while front < end { //for y in R..height-R
                    last = cast_u32(front.read());
                    sum = add(sum, last);
                    front = front.byte_add(source_stride*U16);
                    scatter(transpose_chunk, rows, srl(mul(factor, sum), 16-headroom(R))); // Scattering 32bit words to increasing offsets: high part only overwrites cells before assignment
                    sum = sub(sum, cast_u32(back.read()));
                    back = back.byte_add(source_stride*U16);
                    rows = add(rows, [SIMD::splat(1); 2]);
                }
                for _y in 0..R { //while back < end { // for y in height-R..height
                    sum = add(sum, last);
                    scatter(transpose_chunk, rows, srl(mul(factor, sum), 16-headroom(R))); // Scattering 32bit words to increasing offsets: high part only overwrites cells before assignment
                    sum = sub(sum, cast_u32(back.read()));
                    back = back.byte_add(source_stride*U16);
                    rows = add(rows, [SIMD::splat(1); 2]);
                }
                column = column.byte_add(32*U16);
                rows = add(rows, [SIMD::splat((32-1)*transpose_stride); 2]); // width==stride
            }
        };
        let range = 0..source.size.x;
        let mut transpose = transpose.as_mut();
        std::thread::scope(|s| for thread in std::array::from_fn::<_, THREADS, _>(|thread| {
            let i0 = range.start + (range.end-range.start)*thread as u32/THREADS as u32;
            let i1 = range.start + (range.end-range.start)*(thread as u32+1)/THREADS as u32;
            assert_eq!((i1-i0)%32, 0, "{} {i0} {i1} {} {} {THREADS}", range.start, range.end, source.size);
            let transpose_chunk = transpose.take_mut(i1-i0);
            let thread = std::thread::Builder::new().spawn_scoped(s, move || task(i0, i1, transpose_chunk)).unwrap();
            thread
        }) { thread.join().unwrap(); });
    }
    transpose
}

pub fn transpose_box_convolve<const R: u32, const THREADS : usize>(source: Image<&[u16]>) -> Image<Box<[u16]>> { transpose_box_convolve_scale::<R, THREADS>(source, (1<<(16-headroom(R))) / (R+1+R) as u32) }
pub fn transpose_box_convolve_scale_max<const R: u32, const THREADS : usize>(source: Image<&[u16]>, max: u16) -> Image<Box<[u16]>> { transpose_box_convolve_scale::<R, THREADS>(source, (1<<(32-headroom(R))) / (max as u32*(R+1+R) as u32)) }

pub fn blur<const R: u32, const THREADS : usize>(image: Image<&[u16]>) -> Option<Image<Box<[u16]>>> { 
    let max = max(image.data);
    if max == 0 { return None; }
    let x = transpose_box_convolve_scale_max::<R, THREADS>(image, max);
    Some(transpose_box_convolve::<R, THREADS>(x.as_ref()))
}

pub fn normalize<const R: u32, const THREADS : usize>(image: Image<&[u16]>, threshold: u16) -> Image<Box<[u16]>> {
    let max = max(image.data);
    let x = transpose_box_convolve_scale_max::<R, THREADS>(image.as_ref(), max);
    let mut blur_then_normal = transpose_box_convolve::<R, THREADS>(x.as_ref());
    //let mut blur_then_normal = transpose_box_convolve_slow(transpose_box_convolve_slow(image.as_ref(), R as u32).as_ref(), R as u32);
    assert_eq!(image.stride, blur_then_normal.stride);
    let max = SIMD::splat(max as u32);
    let ref task = |i0, i1, blur_then_normal_chunk:&mut [u16]| unsafe {
        let blur_then_normal = blur_then_normal_chunk.as_mut_ptr();
        let start = image.as_ptr();
        let [image, end] = [i0,i1].map(|i| start.add(i as usize));
        let mut blur_then_normal = blur_then_normal as *mut u16x32;
        let [mut image, end] = [image, end].map(|p| p as *const u16x32);
        let threshold = SIMD::splat(threshold);
        while image < end {
            {
                let low = blur_then_normal.read();
                let low = low.simd_max(threshold).cast::<u32>();
                let image = image.read().cast::<u32>();
                let normal = srl(sll(image, 16)/srl(max*low, 16), 3); // 32bit precision should be enough
                //>>3: rescale to prevent amplified bright high within dark low from clipping (FIXME: fixed rescale either quantize too much or clip)
                //assert!(normal < 0x10000);
                *blur_then_normal = normal.cast::<u16>();
            }
            blur_then_normal = blur_then_normal.add(1);
            image = image.add(1);
        }
    };
    {
        let mut blur_then_normal = blur_then_normal.data.as_mut();
        let range = 0..image.len();
        const THREADS : usize = 4;
        std::thread::scope(|s| std::array::from_fn::<_, THREADS, _>(|thread| {
            let i0 = range.start + (range.end-range.start)/(32*THREADS)*(32*THREADS)*thread/THREADS;
            let i1 = range.start + (range.end-range.start)/(32*THREADS)*(32*THREADS)*(thread+1)/THREADS;
            assert_eq!((i1-i0)%32, 0, "{} {i0} {i1} {}", range.start, range.end);
            let blur_then_normal_chunk = blur_then_normal.take_mut(..(i1-i0)).unwrap();
            let thread = std::thread::Builder::new().spawn_scoped(s, move || task(i0, i1, blur_then_normal_chunk)).unwrap();
            thread
        }).map(|thread| thread.join().unwrap() )); // FIXME: skipping unaligned tail
    }
    blur_then_normal
}

pub fn transpose_box_convolve_slow(source: Image<&[u16]>, R: u32) -> Image<Box<[u16]>> {
    let mut transpose = Image::uninitialized(source.size.yx());
    let MinMax{min, max} = minmax(source.iter().copied()).unwrap();
    if !(min < max) { return transpose;}
    for y in 0..source.size.y {
        let mut sum = (source[xy{x: 0, y}] as u32)*(R as u32);
        for x in 0..R { sum += source[xy{x, y}] as u32; }
        for x in 0..R {
            sum += source[xy{x: x+R, y}] as u32;
            transpose[xy{x: y, y: x}] = ((sum-min as u32*(R+1+R) as u32) as u64*0xFFFF / ((max-min) as u32*(R+1+R) as u32) as u64) as u16;
            sum -= source[xy{x: 0, y}] as u32;
        }
        for x in R..source.size.x-R {
            sum += source[xy{x: x+R, y}] as u32;
            transpose[xy{x: y, y: x}] = ((sum-min as u32*(R+1+R) as u32) as u64*0xFFFF / ((max-min) as u32*(R+1+R) as u32) as u64) as u16;
            sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u32;
        }
        for x in source.size.x-R..source.size.x {
            sum += source[xy{x: source.size.x-1, y}] as u32;
            transpose[xy{x: y, y: x}] = ((sum-min as u32*(R+1+R) as u32) as u64*0xFFFF / ((max-min) as u32*(R+1+R) as u32) as u64) as u16;
            sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u32;
        }
    }
    transpose
}
pub fn blur_slow(image: Image<&[u16]>, R: u32) -> Image<Box<[u16]>> { transpose_box_convolve_slow(transpose_box_convolve_slow(image.as_ref(), R as u32).as_ref(), R as u32) }
pub fn normalize_slow(image: Image<&[u16]>, R: u32, _threshold: u16) -> Option<Image<Box<[u16]>>> {
    let vector::MinMax{min, max} = vector::minmax(image.iter().copied()).unwrap();
    if !(min < max) { return None; }
    let mut blur_then_normal = blur_slow(image.as_ref(), R as u32);
    blur_then_normal.as_mut().zip_map(&image, |&low, &p| {
        let high = (p-min) as u32*0xFFFF/(max-min) as u32;
        assert!(high <= 0xFFFF);
        /*let low = low.max(threshold);
        let div = ((high*0xFFFF/low as u32)>>3) as u16;
        div*/
        (0x8000+high as i32-low as i32).clamp(0,0xFFFF) as u16
    });
    Some(blur_then_normal)
}

pub fn otsu(image: Image<&[u16]>) -> u16 {
    assert!(image.len() < 1<<24);
    let mut histogram : [u32; 0x10000] = [0; 0x10000];
    let mut pixel = 0;
    while pixel < image.len() {
        let read = image[pixel];
        histogram[read as usize] += 1;
        pixel = pixel+1;
    }
    
    type u40 = u64;
    let sum : u40 = histogram.iter().enumerate().map(|(i,&v)| i/*16*/ as u40 * v/*24*/ as u40).sum();
    
    let mut threshold : u16 = 0;
    let mut maximum_variance = 0;
    type u24 = u32;
    let (mut background_count, mut background_sum) : (u24, u40)= (0, 0);
    for (i, &count) in histogram.iter().enumerate() {
        background_count += count;
        if background_count == 0 { continue; }
        if background_count as usize == image.len() { break; }
        background_sum += i/*16*/ as u40 * count/*24*/ as u40;
        let foreground_count : u24 = image.len() as u24 - background_count;
        let foreground_sum : u40 = sum - background_sum;
        type u48 = u64;
        let variance = sq((foreground_sum as u64*background_count as u64 - background_sum as u64*foreground_count as u64) as u128)
                            / (foreground_count as u48*background_count as u48) as u128;
        if variance >= maximum_variance { (threshold, maximum_variance) = (i as u16, variance); }
    }
    threshold
}

fn binary16(image: Image<&[u16]>, threshold: u16, inverse: bool) -> Image<Box<[u16]>> { Image::from_iter(image.size, image.iter().map(|&p| match inverse {
    false => if p>threshold { 0xFFFF } else { 0 },
    true => if p<threshold { 0xFFFF } else { 0 }
}))}

fn binary<const INVERSE: bool>(image: Image<&[u16]>, threshold: u16) -> Image<Box<[u8]>> { 
    let mut target = Image::<Box<[u8]>>::uninitialized(image.size);
    {
        assert_eq!(image.stride, target.stride);
        let Range{start: mut image, end} = image.as_ptr_range();
        let mut target = target.as_mut_ptr();
        while image < end {unsafe{
            *target = if *image > threshold { if INVERSE { 0 } else { 0xFF } } else { if INVERSE { 0xFF } else { 0 } };
            image = image.add(1);
            target = target.add(1);
        }}
    }
    target
}

fn erode<const INVERSE: bool>(high: Image<&[u16]>, erode_steps: usize, threshold: u16) -> Image<Box<[u8]>> {
    let mut binary = self::binary::<INVERSE>(high.as_ref(), threshold);
    let mut erode = Image::zero(binary.size);
    for i in 0..erode_steps {
        for y in 1..erode.size.y-1 {
            for x in 1..erode.size.x-1 {
                let p = |dx, dy| binary[xy{x:(x as i32+dx) as u32,y: (y as i32+dy) as u32}];
                erode[xy{x,y}] = 
                    if i%2 == 0 {[p(0,-1),p(-1,0),p(0,0),p(1,0),p(0,1)].into_iter().max()}
                    else {[p(-1,-1),p(0,-1),p(1,-1),p(-1,0),p(0,0),p(1,0),p(-1,1),p(0,1),p(1,1)].into_iter().max()}.unwrap();
            }
        }
        std::mem::swap(&mut binary, &mut erode);
    }
    binary
}

fn quads(contours: &[imageproc::contours::Contour<u16>], min_area: f32) -> Box<[[vec2; 4]]> {
    let mut quads = Vec::new();
    for contour in contours {
        let contour = contour.points.iter().map(|p| vec2::from(xy{x: p.x, y: p.y})).collect::<Box<_>>();
        if contour.len() < 4 {continue;}
        let quad = |Q:&[vec2]| -> Option<[vec2; 4]> {
            assert!(Q.len() >= 4);
            let mut Q:Vec<_> = Q.iter().copied().collect();
            while Q.len() > 4 {
                Q.remove(((0..Q.len()).map(|i| {
                    let [p0, p1, p2]  = std::array::from_fn(|j| Q[(i+j)%Q.len()]);
                    (cross2(p1-p0, p2-p0), i)
                }).min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap().1+1)%Q.len());
            }
            Some(Q.try_into().unwrap())
        };
        let Some([a,b,c,d]) = quad(&contour) else {continue;};
        let area = {let abc = cross2(b-a,c-a); if abc<0. {continue;} let cda = cross2(d-c,a-c); if cda<0. {continue;} (abc+cda)/2.};
        if area < min_area { continue; }
        if [a,b,c,d,a].array_windows().any(|&[a,b]| vector::sq(b-a)>64.*64./*256.*256.*/) {continue;}
        quads.push([a,b,c,d]);
    }
    quads.into_boxed_slice()
}

/*fn remove_isolated(quads: &[[vec2; 4]], max_distance: f64) -> Box<[[vec2; 4]]> { //32-256
    let all_quads = quads.clone();
    quads.retain(|a| all_quads.iter().any(|b| b!=a && a.iter().any(|p| b.iter().any(|q| vector::sq(q-p) < sq(max_distance)))));
    quads.into_boxed_slice()
}*/

pub fn convex_hull(points: &[vec2]) -> Vec<vec2> {
    let mut p0 = *points.iter().min_by(|a,b| a.x.total_cmp(&b.x)).unwrap();
    let mut Q = vec![];
    loop {
        Q.push(p0);
        let mut next = points[0];
        for &p in points.iter() {
            if next==p0 || cross2(next-p0, p-p0) > 0. { next = p; }
        }
        if next == Q[0] { break; }
        p0 = next;
    }
    Q
}

/*pub fn simplify(mut P: Vec<vec2>) -> Vec<vec2> {
    // Simplifies polygon to 4 corners
    while P.len() > 4 {
        P.remove(((0..P.len()).map(|i| {
            let [p0, p1, p2]  = std::array::from_fn(|j| P[(i+j)%P.len()]);
            (cross2(p2-p0, p1-p0), i)
        }).min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap().1+1)%P.len());
    }
    P
}*/

pub fn simplify(mut P: Vec<vec2>) -> [vec2; 4] {
    // Simplifies polygon to 4 corners
    while P.len() > 4 {
        let (_, [a,b], p) = P.iter().enumerate().map_windows(|[(_a,&A),(b,&B),(c,&C),(_d,&D)]| {
            use vector::xyz;
            let xy1 = |xy{x,y}| xyz{x,y,z:1.};
            let P = vector::cross(xy1(B-A),xy1(C-D));
            if P.z.abs() > 1. {
                let P = xy{x:P.x/P.z, y:P.y/P.z};
                let area = cross2(P-B,C-B).abs()/*FIXME*/;
                assert!(area >= 0., "{A} {B} {C} {D} {P} {area}");
                (area, [*b,*c], P)
            } else {
                (0., [*b,*c], B) // Remove colinear vertices
            }
        }).min_by(|(a,_,_),(b,_,_)| a.total_cmp(b)).unwrap();
        assert_eq!(a+1, b);
        P[a] = p;
        P.remove(b);
    }
    P.try_into().unwrap()
}

pub fn top_left_first(Q: [vec2; 4]) -> [vec2; 4] { let i0 = Q.iter().enumerate().min_by(|(_,a),(_,b)| (a.x+a.y).total_cmp(&(b.x+b.y))).unwrap().0; [0,1,2,3].map(|i|Q[(i0+i)%4]) }
pub fn long_edge_first(mut Q: [vec2; 4]) -> [vec2; 4] { if norm(Q[2]-Q[1])+norm(Q[0]-Q[3]) > norm(Q[1]-Q[0])+norm(Q[3]-Q[2]) { Q.swap(1,3); } Q }

pub fn checkerboard_quad<const INVERSE: bool>(high: Image<&[u16]>, erode_steps: usize, min_side: f32, debug: &'static str) -> Result<[vec2; 4],Image<Box<[u16]>>> {
    let threshold = otsu(high.as_ref());
    if debug=="threshold" { return Err(binary16(high.as_ref(), threshold, INVERSE)); }
    let erode = erode::<INVERSE>(high.as_ref(), erode_steps, threshold);
    //assert_eq!(erode_steps, 0); // Blur is enough
    //let erode = self::binary::<INVERSE>(high.as_ref(), threshold);
    if debug=="erode" { return Err(Image::from_iter(erode.size, erode.iter().map(|&p| (p as u16)<<8))); }
    let ref contours = imageproc::contours::find_contours::<u16>(&imagers::GrayImage::from_vec(erode.size.x, erode.size.y, erode.data.into_vec()).unwrap());
    if debug=="contour" { let mut target = Image::zero(erode.size); for contour in contours { for p in &contour.points { target[xy{x: p.x as u32, y: p.y as u32}] = 0xFFFF; }} return Err(target); }
    let quads = quads(contours, sq(min_side)); // 0.2s
    /*let connectivity = vec![0; quads.len()*quads.len()];
    for (i,P) in quads.iter().enumerate() {
        for &p in P {
            let (_,[a,b],j) = quads.iter().enumerate().filter(|&(j,_)| i!=j).map(|(j,Q)| Q.iter().map(move |&q| (vector::sq(q-p),[q,p],j))).flatten().min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap();
            if a!=b && vector::sq(a-b)<32.*32. { connectivity[i*quads.len()+j] = 1; }
        }
    }*/
    let all_quads = quads.clone(); // debug=="quads"
    let largest_component = {
        let mut quads = quads.into_vec();
        let mut largest_component = vec![];
        while let Some(P) = quads.pop() {
            let mut component = vec![P];
            let mut queue = vec![P];
            while let Some(P) = queue.pop() {
                for p in P {
                    if let Some((_,[a,b],j)) = quads.iter().enumerate().map(|(j,Q)| Q.iter().map(move |&q| (vector::sq(q-p),[q,p],j))).flatten().min_by(|(a,_,_),(b,_,_)| a.total_cmp(b)) {
                        if a!=b && vector::sq(a-b)<32.*32. { let Q = quads.swap_remove(j); component.push(Q); queue.push(Q) }
                    }
                }
            }
            if component.len() > largest_component.len() { largest_component = component; }
        }
        largest_component
    };
    if debug=="quads" { let mut target = Image::zero(erode.size);
        for q in &*all_quads { for [&a,&b] in {let [a,b,c,d] = q; [a,b,c,d,a]}.array_windows() { for (p,_,_,_) in ui::line::generate_line(target.size, [a,b]) { target[p] = 0x8000; } } }
        for q in &*largest_component { for [&a,&b] in {let [a,b,c,d] = q; [a,b,c,d,a]}.array_windows() { for (p,_,_,_) in ui::line::generate_line(target.size, [a,b]) { target[p] = 0xFFFF; } } }
        /*for P in &*quads { 
            let (_,[a,b]) = quads.iter().filter(|&Q| Q!=P).map(|Q| P.iter().map(|&p| Q.iter().map(move |&q| (vector::sq(q-p),[q,p])))).flatten().flatten().min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap();
            if a!=b { for (p,_,_,_) in ui::line::generate_line(target.size, [a,b]) { target[p] = 0xFFFF; } }
        }*/
        /*for P in &*quads {
            for &p in P {
                let (_,[a,b]) = quads.iter().filter(|&Q| Q!=P).map(|Q| Q.iter().map(move |&q| (vector::sq(q-p),[q,p]))).flatten().min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap();
                if a!=b && vector::sq(a-b)<32.*32. { for (p,_,_,_) in ui::line::generate_line(target.size, [a,b]) { target[p] = 0x8000; } }
            }
        }*/
        return Err(target); 
    }
    let quads = largest_component.into_boxed_slice();
    let points : Box<_> = quads.into_iter().map(|q| q.into_iter()).flatten().copied().collect();
    if points.len() < 4 { return Err(binary16(high.as_ref(), threshold, false)); }
    let Q = simplify(convex_hull(&*points));
    let Ok(Q) = Q.try_into() else { return Err(binary16(high, threshold, false)) };
    Ok(long_edge_first(top_left_first(Q)))
}

pub fn cross_response<const R: u32, const THREADS: usize>(high: Image<&[u16]>) -> Option<Image<Box<[u16]>>> {
    let mut cross = Image::zero(high.size);
    for y in R..high.size.y-R {
        for x in R..high.size.x-R {
            let [p00,p10,p01,p11] = {let r=R as i32;[xy{x:-r,y:-r},xy{x:r,y:-r},xy{x:-r,y:r},xy{x:r,y:r}]}.map(|d|high[(xy{x,y}.signed()+d).unsigned()]);
            let threshold = ([p00,p10,p01,p11].into_iter().map(|u16| u16 as u32).sum::<u32>()/4) as u16;
            if p00<threshold && p11<threshold && p10>threshold && p01>threshold ||
                p00>threshold && p11>threshold && p10<threshold && p01<threshold {
                cross[xy{x,y}] = (num::abs(p00 as i32 + p11 as i32 - (p10 as i32 + p01 as i32))/2) as u16;
            }
        }
    }
    //Some(cross)
    Some(blur_slow(cross.as_ref(), 1))
    //blur::<R, THREADS>(cross.as_ref())
}

pub fn checkerboard_direct_intersections<const R: u32, const THREADS: usize>(high: Image<&[u16]>, _max_distance: u32, debug: &'static str) -> std::result::Result<Vec<uint2>,Image<Box<[u16]>>> {
    let Some(cross) = self::cross_response::<R, THREADS>(high.as_ref()) else { return Err(high.clone()) };
    if debug=="response" { return Err(cross); }
    
    let threshold = otsu(cross.as_ref()); //foreground_mean
    if debug=="threshold" { return Err(Image::from_iter(cross.size, cross.iter().map(|&p| if p>threshold { p } else { 0 }))); }

    let mut points = Vec::new();
    const R : u32 = 12;
    for y in R..cross.size.y-R { for x in R..cross.size.x-R {
        let center = cross[xy{x, y}];
        if center < threshold { continue; }
        let mut flat = 0;
        if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            if (dx,dy) == (0,0) { continue; }
            if cross[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
            if cross[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
        }} true})() { points.push(xy{x,y}); }
    }}
    if points.len() < 4 { return Err(cross); }
    /*let points : Vec<_> = points.iter().map(|&a| (a, {
        let mut p = points.iter().filter(|&b| a!=*b).copied().collect::<Box<_>>();
        assert!(p.len() >= 3);
        let closest = if p.len() > 3 { let (closest, _, _) = p.select_nth_unstable_by_key(3, |p| vector::sq(p-a)); &*closest } else { &p };
        <[uint2; 3]>::try_from(closest).unwrap().map(|p| (vector::sq(p-a)<sq(max_distance)).then_some(p))
    })).collect();

    Ok(points.iter().map(|&p| {
        let mut connected = Vec::new();
        fn walk(points: &[(uint2, [Option<uint2>; 3])], mut connected: &mut Vec<uint2>, (p, neighbours): (uint2, [Option<uint2>; 3])) {
            if !connected.contains(&p) {
                connected.push(p);
                for p in neighbours.iter().filter_map(|p| *p) { walk(points, &mut connected, *points.iter().find(|(q,_)| q==&p).unwrap()); }
            }
        }
        walk(&points, &mut connected, p);
        connected
    }).max_by_key(|g| g.len()).unwrap())*/
    Ok(points)
}

pub fn refine<const R: u32, const THREADS: usize>(high: Image<&[u16]>, mut points: [vec2; 4], radius: u32, debug: &'static str) -> Result<[vec2; 4],(vec2, Box<[vec2]>, Box<[vec2]>, Image<Box<[uint2]>>, vec2, vec2, Image<Box<[u16]>>)> {
    let cross = cross_response::<R, THREADS>(high.as_ref()).unwrap();

    for _ in 0..1 {
        let grid = {
            let cross = &cross;
            Image::from_iter(xy{x:6,y:4},
          (1..=4).map(|y|
                    (1..=6).map(move |x| {
                        let xy{x, y} = xy{x: x as f32/7., y: y as f32/5.};
                        let [A,B,C,D] = points.clone();
                        let p = y*(x*A+(1.-x)*B) + (1.-y)*(x*D+(1.-x)*C);
                        let p = xy{x: p.x.round() as u32, y: p.y.round() as u32,};
                        let (p,_) = (p.y-radius..p.y+radius).map(|y| (p.x-radius..p.x+radius).map(move |x| xy{x,y})).flatten().map(|p| (p, cross[p])).max_by_key(|(_,v)| *v).unwrap_or((p,0));
                        p
                    }
                    )
                ).flatten()
            )
        };
        let ref ref_grid = grid;
        let rows = (0..grid.size.y).map(|y| (0..grid.size.x).map(move |x| ref_grid[xy{x,y}]));
        let columns = (0..grid.size.x).map(|x| (0..grid.size.y).map(move|y| ref_grid[xy{x,y}]));
        let column = rows.map(|row| row.map(|p| vec2::from(p)).sum::<vec2>()/6.).collect::<Box<_>>();
        let row = columns.map(|column| column.map(|p| vec2::from(p)).sum::<vec2>()/4.).collect::<Box<_>>();
        let row_axis = row.last().unwrap()-row[0];
        let column_axis = column.last().unwrap()-column[0];
        let center = grid.iter().map(|&p| vec2::from(p)).sum::<vec2>()/(6.*4.);
        if debug != "" { return Err((center, row, column, grid, row_axis, column_axis, cross)); }
        points = [xy{x:-1./2.,y:-1./2.},xy{x:1./2.,y:-1./2.},xy{x:1./2.,y:1./2.},xy{x:-1./2.,y:1./2.}].map(|xy{x,y}| center + x*row_axis + y*column_axis);
    }
    Ok(points)
}

pub fn cross(target: &mut Image<&mut[u32]>, scale:f32, offset:uint2, p:vec2, color:u32) {
    let mut plot = |dx,dy| {
        let Some(p) = (int2::from(scale*p)+xy{x: dx, y: dy}).try_unsigned() else {return};
        if let Some(p) = target.get_mut(offset+p) { *p = color; }
    };
    for dy in -64..64 { plot(0, dy); }
    for dx in -64..64 { plot(dx, 0); }
}

pub fn checkerboard_quad_debug<const INVERSE: bool>(nir: Image<&[u16]>, erode_steps: usize, min_side: f32, debug: &'static str, target: &mut Image<&mut [u32]>) -> Option<[vec2; 4]> { 
    use super::image::scale;
    match checkerboard_quad::<INVERSE>(nir.as_ref(), erode_steps, min_side, debug) {
        Err(image) => { scale(target, image.as_ref()); None }
        Ok(points) => {
            /*let points = match refine(nir.as_ref(), points, 0/*128*/, debug) {
                Err((center, row, column, grid, row_axis, column_axis, corner)) => {
                    let (_, scale, offset) = scale(target, corner.as_ref());
                    cross(target, scale, offset, center, 0xFFFFFF);
                    for x in 0..grid.size.x {
                        cross(target, scale, offset, row[x as usize], 0x00FF00);
                    }
                    for y in 0..grid.size.y {
                        cross(target, scale, offset, column[y as usize], 0x00FF00);
                        for x in 0..grid.size.x {
                            cross(target, scale, offset, grid[xy{x,y}].into(), 0x00FFFF);
                            let xy{x, y} = xy{x: x as f32/7.,  y: y as f32/5.};
                            let xy{x, y} = xy{x: x-1./2., y: y-1./2.};
                            let p = center + x*row_axis + y*column_axis;
                            cross(target, scale, offset, p, 0xFF00FF); // purple
                        }
                    }
                    return None;
                }
                Ok(points) => points
            };*/
            if debug=="z" {
                /*let (_, scale, offset) =*/ scale(target, nir.as_ref());
                //for (i,&p) in points.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
                return None;
            }
            Some(points)
        }
    }
}

pub fn fit_rectangle(points: &[uint2]) -> [vec2; 4] {
    let area = |[a,b,c,d]:[vec2; 4]| {let abc = vector::cross2(b-a,c-a); let cda = vector::cross2(d-c,a-c); (abc+cda)/2.};
    use std::f32::consts::PI;
    [0./*,PI/2.*/].map(|_angle| {
        // TODO: rotate
        let vector::MinMax{min, max} = vector::minmax(points.iter().map(|p| p.map(|u32| u32 as f32))).unwrap();    
        [xy{x: min.x, y: min.y}, xy{x: max.x, y: min.y}, xy{x: max.x, y: max.y}, xy{x: min.x, y: max.y}]
        // TODO: rotate back
    }).into_iter().min_by(|&a,&b| f32::total_cmp(&area(a), &area(b))).unwrap()
}