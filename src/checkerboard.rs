use {std::cmp::{min,max}, num::{sq,zero}, vector::{xy, uint2, int2, vec2, cross2, norm}, image::Image};
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
    let binary = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFF } else { 0 }));
    //([vec2::from(binary.size/2); 4], binary)

    /*// Distance
    let distance = {
        const R : u32 = 256;
        let mut target = Image::uninitialized(binary.size-uint2::from(2*R));
        for y in R..binary.size.y-R {
            for x in R..binary.size.x-R {
                let r = (||{ for r in 1..R {
                    let circle = std::iter::from_generator(|| {
                        for x in x-r..x+r { yield xy{x,y: y-r}; yield xy{x,y: y+r}; }
                        for y in y-r+1..y+r-1 { yield xy{x: x-r, y}; yield xy{x: x-r, y}; }
                    });
                    for p in circle { if binary[p] == 0 { return r } }
                } /*unreachable!()*/return R-1;})();
                target[xy{x: x-R, y: y-R}] = r as u8;
            }
        }
        target
    };
    ([vec2::from(distance.size/2); 4], distance)*/

    trait ToDistance { fn to_distance(&self) -> u32; }
    impl ToDistance for u32 { fn to_distance(&self) -> u32 { *self } }
    impl ToDistance for u8 { fn to_distance(&self) -> u32 { if *self > 0 { u32::MAX } else { 0 } } }
    fn transpose_distance_transform_1D<T:ToDistance>(f: &[T], mut target: Image<&mut [u32]>, x: u32) {
        let mut v : Box<[u32]> = unsafe{Box::new_uninit_slice(f.len()).assume_init()};
        v[0] = 0;
        let mut z = unsafe{Box::new_uninit_slice(f.len()+1).assume_init()};
        z[0] = f32::NEG_INFINITY;
        z[1] = f32::INFINITY;

        let mut k = 0;
        for q in 1..f.len() as u32 {
            let mut s = ((f[q as usize].to_distance()+sq(q)) as i32 - (f[v[k] as usize].to_distance()+sq(v[k])) as i32) as f32 / (2*q-2*v[k]) as f32;
            while s <= z[k] {
                k -= 1;
                s = ((f[q as usize].to_distance()+sq(q)) as i32 - (f[v[k] as usize].to_distance()+sq(v[k])) as i32) as f32 / (2*q-2*v[k]) as f32;
            }
            k += 1;
            v[k] = q as u32;
            z[k] = s;
            z[k+1] = f32::INFINITY;
        }

        let mut k = 0;
        for q in 0..f.len() as u32 {
            while z[k+1] <= q as f32 { k += 1; }
            target[xy{x, y: q}] = sq(q-v[k]) + f[v[k] as usize].to_distance();
        }
    }

    fn distance_transform(binary: Image<&[u8]>) -> Image<Box<[u32]>> {
        let mut transpose = Image::uninitialized(binary.size.yx());
        for (y, row) in binary.enumerate() { transpose_distance_transform_1D(row, transpose.as_mut(), y as u32); }
        let mut distance = Image::uninitialized(transpose.size.yx());
        for (y, row) in transpose.as_ref().enumerate() { transpose_distance_transform_1D(row, distance.as_mut(), y as u32); }
        //panic!("{}", distance.iter().max().unwrap());
        distance
    }
    let distance = distance_transform(binary.as_ref());
    ([vec2::from(distance.size/2); 4], Image::from_iter(distance.size, distance.iter().map(|&v| v.min(0xFF) as u8)))

    /*let max = {
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

    let mut max = max;
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
