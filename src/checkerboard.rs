use {/*std::cmp::{min,max},*/ num::{sq/*,zero*/}, vector::{xy, /*uint2, int2,*/ vec2/*, cross2, norm*/}, image::Image};
pub fn checkerboard(image: Image<&[u8]>) -> ([vec2; 4], Image<Box<[u8]>>) {
    /*// High pass
    fn transpose_low_pass_1D(source: Image<&[u8]>) -> Image<Box<[u8]>> {
        let mut transpose = Image::uninitialized(source.size.yx());
        const R : u32 = 127;
        //const factor : u32 = 0x100 / (R+1+R) as u16;
        for y in 0..source.size.y {
            let mut sum = (source[xy{x: 0, y}] as u16)*(R as u16);
            for x in 0..R { sum += source[xy{x, y}] as u16; }
            for x in 0..R {
                sum += source[xy{x: x+R, y}] as u16;
                transpose[xy{x: y, y: x}] = ((sum /* * factor*/) >> 8) as u8;
                sum -= source[xy{x: 0, y}] as u16;
            }
            for x in R..source.size.x-R {
                sum += source[xy{x: x+R, y}] as u16;
                transpose[xy{x: y, y: x}] = ((sum /* * factor*/) >> 8) as u8;
                sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u16;
            }
            for x in source.size.x-R..source.size.x {
                sum += source[xy{x: source.size.x-1, y}] as u16;
                transpose[xy{x: y, y: x}] = ((sum /* * factor*/) >> 8) as u8;
                sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u16;
            }
        }
        transpose
    }
    let source = image;
    let mut low_then_high = transpose_low_pass_1D(transpose_low_pass_1D(source.as_ref()).as_ref());
    low_then_high.as_mut().zip_map(&source, |&low, &p| (128+(p as i16-low as i16).clamp(-128,127)) as u8);
    let high_pass = low_then_high;
    ([vec2::from(source.size/2); 4], high_pass)
    let image = high_pass.as_ref();*/

    assert!(image.len() < 1<<24);
    let mut histogram : [u32; 256] = [0; 256];
    for &pixel in image.iter() { histogram[pixel as usize] += 1; }
    let sum : u32 = histogram.iter().enumerate().map(|(i,&v)| i/*8*/ as u32 * v/*24*/ as u32).sum();
    let mut threshold : u8 = 0;
    let mut maximum_variance = 0;
    type u24 = u32;
    let (mut background_count, mut background_sum) : (u24, u32)= (0, 0);
    for (i, &count) in histogram.iter().enumerate() {
        background_count += count;
        if background_count == 0 { continue; }
        if background_count as usize == image.len() { break; }
        background_sum += i/*8*/ as u32 * count/*24*/ as u32;
        let foreground_count : u24 = image.len() as u24 - background_count;
        let foreground_sum : u32 = sum - background_sum;
        type u48 = u64;
        let variance = sq((foreground_sum as u64*background_count as u64 - background_sum as u64*foreground_count as u64) as u128)
                            / (foreground_count as u48*background_count as u48) as u128;
        if variance >= maximum_variance { (threshold, maximum_variance) = (i as u8, variance); }
    }
    //let binary = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFF } else { 0 }));
    //return ([vec2::from(binary.size/2); 4], binary);

    fn distance(image: Image<&[u8]>, threshold: u8, inverse: bool) -> Image<Box<[u8]>> {
        let size = image.size;
        let mut G = Image::uninitialized(size);
        for x in 0..size.x {
            G[xy{x, y: 0}] = if (image[xy{x, y: 0}] > threshold) ^ inverse { 0 } else { size.x+size.y };
            for y in 1..size.y {
                G[xy{x, y}] = if (image[xy{x, y}] > threshold) ^ inverse { 0 } else { 1+G[xy{x, y: y-1}] };
            }
            for y in (0..size.y-1).rev() {
                if G[xy{x, y: y+1}] < G[xy{x, y}] {
                    G[xy{x, y}] = 1 + G[xy{x, y: y+1}];
                }
            }
        }
        let mut S = unsafe{Box::new_uninit_slice(size.x as usize).assume_init()};
        let mut T = unsafe{Box::new_uninit_slice(size.x as usize).assume_init()};
        let mut distance = Image::uninitialized(size);
        for y in 0..size.y {
            let mut q = 0i16;
            S[0] = 0;
            T[0] = 0;
            let g = |i| G[xy{x: i as u32, y}];
            let f = |x,i| sq((x as i16 - i as i16) as i32) as u32 + sq(g(i) as u32);
            for u in 1..size.x as u16 {
                while q >= 0 && f(T[q as usize], S[q as usize]) > f(T[q as usize], u) { q = q - 1; }
                if q < 0 { q = 0; S[0] = u; }
                else {
                    let Sep = |i,u| (sq(u as u32) - sq(i as u32) + sq(g(u) as u32) - sq(g(i) as u32)) / (2*(u as i16 - i as i16) as u32);
                    let w = 1 + Sep(S[q as usize], u);
                    if w < size.x { 
                        q = q + 1;
                        S[q as usize] = u;
                        T[q as usize] = w as u16;
                    }
                }
            }
            for u in (0..size.x as u16).rev() {
                let d = f(u, S[q as usize]);
                distance[xy{x: u as u32, y}] = if d > 0x2000 { 0 } else { (d*0xFF/0x2000) as u8 }; // /((.sqrt()).min(255.) as u8;
                //distance[xy{x: u, y}] = f(u,S[q as usize]).min(0xFF) as u8;
                if T[q as usize] == u { q = q - 1; }
            }
        }
        distance
    }
    let black = distance(image.as_ref(), threshold, false);
    //let white = distance(image.as_ref(), threshold, true);
    //return ([vec2::from(image.size/2); 4], black);
    let distance = black;
    
    let max = {
        const R : u32 = 2;
        let mut target = Image::zero(distance.size);
        for y in R..distance.size.y-R { for x in R..distance.size.x-R {
            let center = distance[xy{x, y}];
            let mut flat = 0;
            if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                if (dx,dy) == (0,0) { continue; }
                if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
                if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
            }} true})() { if flat < 12 { target[xy{x,y}] = center; } }
        }}
        target
    };
    return ([vec2::from(image.size/2); 4], max)

    /*let mut max = max;
    let mut points = Vec::new();
    {const R : u32 = 4; // Merges close peaks (top left first)
    for y in R..max.size.y-R { for x in R..max.size.x-R {
        let value = max[xy{x,y}];
        if value == 0 { continue; }
        let mut flat = 0;
        let mut sum : uint2 = zero();
        for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
            if max[p] == 0 { continue; }
            flat += 1;
            sum += p;
        }}
        points.push( (vec2::from(int2::from(xy{x,y}+sum))/((1+flat) as f32), value) );
        for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
            max[p] = 0; // Merge small flat plateaus as single point
        }}
    }}}

    let (points, _, _) = points.select_nth_unstable_by(4*5+3*4, |(_,a), (_,b)| b.cmp(a));
    let mut p0 = points.iter().min_by(|a,b| a.0.x.total_cmp(&b.0.x)).unwrap().0;
    let mut Q = vec![];
    loop {
        Q.push(p0);
        let mut next = points[0].0;
        for &(p, _) in points.iter() {
            if next==p0 || cross2(next-p0, p-p0) > 0. { next = p; }
        }
        if next == Q[0] { break; }
        p0 = next;
    }

     // Simplifies polygon to 4 corners
    while Q.len() > 4 {
        Q.remove(((0..Q.len()).map(|i| {
            let [p0, p1, p2]  = std::array::from_fn(|j| Q[(i+j)%Q.len()]);
            (cross2(p2-p0, p1-p0), i)
        }).min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap().1+1)%Q.len());
    }

    // First edge is long edge
    if norm(Q[2]-Q[1])+norm(Q[0]-Q[3]) > norm(Q[1]-Q[0])+norm(Q[3]-Q[2]) { Q.swap(0,3); }
    (Q.try_into().unwrap(), high_pass)*/
}
