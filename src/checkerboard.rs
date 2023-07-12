use {num::{sq, zero}, vector::{xy, uint2, int2, vec2, cross2, norm}, image::Image};
#[allow(dead_code)] pub enum Result {
    Checkerboard([vec2; 4]),
    Image(Image<Box<[u16]>>),
    Points([Vec<(vec2, u16)>; 2], Image<Box<[u16]>>),
}
pub fn checkerboard(image: Image<&[u16]>, toggle: bool) -> Result {
    fn transpose_low_pass_1D<const R: u32>(source: Image<&[u16]>) -> Image<Box<[u16]>> {
        let mut transpose = Image::uninitialized(source.size.yx());
        /*const*/let factor : u32 = 0x1000 / (R+1+R) as u32;
        for y in 0..source.size.y {
            let mut sum = (source[xy{x: 0, y}] as u32)*(R as u32);
            for x in 0..R { sum += source[xy{x, y}] as u32; }
            for x in 0..R {
                sum += source[xy{x: x+R, y}] as u32;
                transpose[xy{x: y, y: x}] = ((sum * factor) >> 12) as u16;
                sum -= source[xy{x: 0, y}] as u32;
            }
            for x in R..source.size.x-R {
                sum += source[xy{x: x+R, y}] as u32;
                transpose[xy{x: y, y: x}] = ((sum * factor) >> 12) as u16;
                sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u32;
            }
            for x in source.size.x-R..source.size.x {
                sum += source[xy{x: source.size.x-1, y}] as u32;
                transpose[xy{x: y, y: x}] = ((sum * factor) >> 12) as u16;
                sum -= source[xy{x: (x as i32-R as i32) as u32, y}] as u32;
            }
        }
        transpose
    }
    //let image = transpose_low_pass_1D::<11>(transpose_low_pass_1D::<11>(image.as_ref()).as_ref());

    // High pass
    let source = image;
    let mut low_then_high = transpose_low_pass_1D::<32>(transpose_low_pass_1D::<32>(source.as_ref()).as_ref());
    low_then_high.as_mut().zip_map(&source, |&low, &p| (0x8000+(p as i32-low as i32).clamp(-0x8000,0x7FFF)) as u16);
    //return Result::Image(low_then_high);
    let high_pass = low_then_high;
    let image = high_pass.as_ref();

    assert!(image.len() < 1<<24);
    let mut histogram : [u32; 65536] = [0; 65536];
    for &pixel in image.iter() { histogram[pixel as usize] += 1; }
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
    let binary = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFFFF } else { 0 }));
    //return Result::Image(binary);

    fn distance<T: TryFrom<u32>>(image: Image<&[u16]>, threshold: u16, inverse: bool) -> Image<Box<[T]>> where T::Error: std::fmt::Debug {
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
                distance[xy{x: u as u32, y}] = d.try_into().unwrap(); //if d > 0x2000 { 0 } else { (d*0xFF/0x2000) as u8 }; // /~32
                if T[q as usize] == u { q = q - 1; }
            }
        }
        distance
    }

    let mut points = [false,true].map(|inverse| { 
        let distance = distance(image.as_ref(), threshold, inverse);
        //return Result::Image(distance)

        let max = {
            const R : u32 = 2;
            let mut target = Image::zero(distance.size);
            for y in R..distance.size.y-R { for x in R..distance.size.x-R {
                let center = distance[xy{x, y}];
                if center < 4*4/*64*64*/ { continue; }
                let mut flat = 0;
                if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                    if (dx,dy) == (0,0) { continue; }
                    if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
                    if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
                }} true})() { if flat < 12 { target[xy{x,y}] = center; } }
            }}
            target
        };
        //return Result::Image(max);

        let mut max = max;
        let mut points = Vec::new();
        {const R : u32 = 8/*64*/; // Merges close peaks (top left first)
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
        points
    });

    /*let points : [Vec<_>; _] = {let [a,b]=points.each_ref(); [[a,b],[b,a]]}.map(|[a,b]| {
        a.into_iter().filter(|a| b.iter().any(|b| true/*vector::sq(a.0-b.0) < 3. * a.1 as f32*/)).copied().collect()
    });*/
    //println!("{}", points[1].len());
    if toggle { 
        for i in 0..4 {
            points[i%2] = points[i%2].iter().filter(|a| points[[1,0][i%2]].iter().any(|b| vector::sq(a.0-b.0) < 6. * a.1 as f32)).copied().collect();
        }
        println!("{} {}", points[0].len(), points[1].len(), points.);
    }
    //println!("{}", points[1].len());
    
    const N : usize = 4*5+3*4;
    //let [_, mut points] = points;
    if points.len() < N { return Result::Points(points, high_pass); }
    return Result::Points(points, high_pass); 

    //let points = if points.len() > N { let (points, _, _) = points.select_nth_unstable_by(N, |(_,a), (_,b)| b.cmp(a)); points } else { points.as_mut_slice() };
    //return Result::Points(points.iter().copied().collect(), high_pass);
    /*let mut p0 = points.iter().min_by(|a,b| a.0.x.total_cmp(&b.0.x)).unwrap().0;
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
    Ok(Q.try_into().unwrap())*/
}
