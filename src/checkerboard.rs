use {num::{sq, zero}, vector::{xy, uint2, int2, vec2, cross2, norm, minmax, MinMax}, image::Image};
pub enum Result {
    Checkerboard([vec2; 4]),
    Image(Image<Box<[u16]>>),
    Points([Vec<(vec2, [Option<vec2>; 4])>; 2], Image<Box<[u16]>>),
}
pub fn checkerboard(image: Image<&[u16]>, black: bool, debug: &'static str) -> Result {
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
    // Low pass
    let low_pass = (image.size.y > 192).then(|| transpose_low_pass_1D::<12>(transpose_low_pass_1D::<12>(image.as_ref()).as_ref()));
    let image = low_pass.as_ref().map(|image| image.as_ref()).unwrap_or(image);
    
    // High pass
    let source = image;
    let mut low_then_high = if source.size.y <= 192 {
        transpose_low_pass_1D::<32>(transpose_low_pass_1D::<32>(source.as_ref()).as_ref())
    } else {
        transpose_low_pass_1D::<127>(transpose_low_pass_1D::<127>(source.as_ref()).as_ref())
    };
    if debug=="low" { return Result::Image(low_then_high); }
    low_then_high.as_mut().zip_map(&source, |&low, &p| (0x8000+(p as i32-low as i32).clamp(-0x8000,0x7FFF)) as u16);
    if debug=="high" { return Result::Image(low_then_high); }
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
    #[allow(unused_variables)] let binary = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFFFF } else { 0 }));
    if debug=="binary" { return Result::Image(binary); }

    let image8 = png::GrayImage::from_vec(image.size.x, image.size.y, image.iter().map(|u16| (u16>>8) as u8).collect()).unwrap();
    let contours = imageproc::contours::find_contours_with_threshold::<u16>(&image8, (threshold>>8) as u8);
    let mut contour_image = Image::zero(image.size);
    for contour in contours {
        for p in contour.points { contour_image[xy{x: p.x as u32, y: p.y as u32}] = 0xFFFF; }
    }
    return Result::Image(contour_image);

    /*fn distance<T: TryFrom<u32>>(image: Image<&[u16]>, threshold: u16, inverse: bool) -> Image<Box<[T]>> where T::Error: std::fmt::Debug {
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

    let mut points = match [false,true].try_map(|inverse| -> std::result::Result<_, Image<Box<[u32]>>> {
        let distance = distance::<u32>(image.as_ref(), threshold, inverse);
        if inverse!=black && debug=="distance" { return Err(distance) }

        let max = {
            const R : u32 = 2;
            let mut target = Image::zero(distance.size);
            for y in R..distance.size.y-R { for x in R..distance.size.x-R {
                let center = distance[xy{x, y}];
                if center < 4*4/*if distance.size.y <= 192 { 4*4 } else { 64*64 }*/ { continue; }
                let mut flat = 0;
                if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
                    if (dx,dy) == (0,0) { continue; }
                    if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
                    if distance[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
                }} true})() { /*if flat < 12*/ { target[xy{x,y}] = center; } }
            }}
            target
        };
        if inverse!=black && debug=="max" { return Err(max) }

        let mut max = max;
        let mut points = Vec::new();
        let R : u32 = if max.size.y <= 192 { 8 } else { 112 }; // Merges close peaks (top left first)
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
        }}
        Ok(points)
    }) {
        Err(image) => return Result::Image({
            let MinMax{min, max} = minmax(image.iter().copied()).unwrap();
            if min<max { Image::from_iter(image.size, image.iter().map(|u| ((u-min) as u64*0xFFFF/(max-min) as u64) as u16)) }
            else { binary }
        }),
        Ok(points) => points
    };
    if debug=="peaks" { return Result::Points(points.map(|points| points.iter().map(|&(p, _)|(p, [None; 4])).collect()), high_pass); }

    //for points in &points { assert!(!points.is_empty()); }
    for points in &points { if points.is_empty() { return Result::Image(high_pass); } }

    let black = if black { 0 } else { 1 };
    const N : usize = 4*5+3*4;
    let all_points = points.clone();
    loop {
        let isolation = [0,1].map(|i| minmax(points[i%2].iter().map(|a| points[[1,0][i%2]].iter().map(|b| vector::sq(a.0-b.0)).min_by(f32::total_cmp).unwrap())).unwrap());
        let (i, isolation) = isolation.into_iter().enumerate().max_by(|(_, a), (_, b)| a.max.total_cmp(&b.max)).unwrap();
        if isolation.max < 5./4.*isolation.min { break; }
        let before = points[i%2].len(); //points.each_ref().map(Vec::len);
        points[i%2] = points[i%2].iter().filter(|a| points[[1,0][i%2]].iter().any(|b| vector::sq(a.0-b.0) < isolation.max/*6. * a.1 as f32*/)).copied().collect();
        //assert_ne!(before, points.each_ref().map(Vec::len));
        assert_ne!(before, points[i%2].len());
        if i==black && points[black].len() == N { break; }
        //if points[i%2].len() ==
        //if points.each_ref().map(Vec::len) != before { assert!(points[i%2].len() == before[i%2]-1); continue; } else { break; }
    }

    return Result::Points([0,1].map(|i| {
        points[i].iter().map(|&(p, _)|
            (p, [None; 4])
        ).collect()
    }), high_pass);*/

    /*if points[black].len() < N { return Result::Points(points, high_pass); }
    if debug=="filtered" { return Result::Points(points, high_pass); }

    let points = &mut points[black]; //  IR "white" = Visible black
    let points = if points.len() > N { let (points, _, _) = points.select_nth_unstable_by(N, |(_,a), (_,b)| b.cmp(a)); points } else { points.as_mut_slice() };
    if debug=="selection" { return Result::Points(if black==1 {[Vec::new(), points.iter().copied().collect()]}else{[points.iter().copied().collect(),Vec::new()]}, high_pass); }

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
    //if toggle && black==1 { return Result::Points(if black==1 {[Vec::new(), Q.iter().map(|&p| (p,0)).collect()]}else{[Q.iter().map(|&p| (p,0)).collect(), Vec::new()]}, high_pass); }
    Result::Checkerboard(Q.try_into().unwrap())*/
}
