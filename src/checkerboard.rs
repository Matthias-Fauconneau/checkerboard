use {std::cmp::{min,max}, num::{sq,zero}, vector::{xy, uint2, int2, vec2, cross2, norm}, image::Image};
pub fn checkerboard(image: Image<&[u8]>) -> [vec2; 4] {
    // High pass
    let mut transpose = Image::zero(image.size.yx());
    const R : u32 = 8;
    let ref source = image;
    for y in 0..source.size.y {
        let mut sum = (source[xy{x:0, y}] as u16)*(R as u16);
        for x in 0..R { sum += source[xy{x, y}] as u16; }
        for x in 0..source.size.x {
            sum += source[xy{x: min(x+R, source.size.x-1), y}] as u16;
            transpose[xy{x: y, y: x}] = sum;
            sum -= source[xy{x: max(0, x as i32-R as i32) as u32, y}] as u16;
        }
    }
    let ref source = transpose;
    let mut transpose = Image::zero(image.size);
    for y in 0..source.size.y {
        let mut sum = (source[xy{x:0, y}] as u32)*(R as u32);
        for x in 0..R { sum += source[xy{x, y}] as u32; }
        for x in 0..source.size.x {
            sum += source[xy{x: min(x+R, source.size.x-1), y}] as u32;
            transpose[xy{x: y, y: x}] = (sum / sq(R+1+R) as u32) as u8;
            sum -= source[xy{x: max(0, x as i32-R as i32) as u32, y}] as u32;
        }
    }
    transpose.as_mut().zip_map(&image, |&low, &p| (128+(p as i16-low as i16).clamp(-128,127)) as u8);
    let image = transpose;

    assert!(image.len() < 1<<16);
    let mut histogram : [u16; 256] = [0; 256];
    for &pixel in image.iter() { histogram[pixel as usize]+=1; }
    type u24 = u32;
    let sum : u24 = histogram.iter().enumerate().map(|(i,&v)| i/*8*/ as u24 * v/*16*/ as u24).sum();
    let mut threshold : u8 = 0;
    let mut maximum_variance = 0.;
    let (mut background_count, mut background_sum) : (u16, u32)= (0, 0);
    for (i, &count) in histogram.iter().enumerate() {
        background_count += count;
        if background_count == 0 { continue; }
        if background_count as usize == image.len() { break; }
        background_sum += i/*8*/ as u24 * count/*16*/ as u24;
        let foreground_count : u16 = image.len() as u16 - background_count;
        let foreground_sum : u24 = sum - background_sum;
        type u40 = u64;
        let variance = sq((foreground_sum as u40*background_count as u40 - background_sum as u40*foreground_count as u40) as f64)
                            /   (foreground_count as u32*background_count as u32) as f64;
        if variance >= maximum_variance { (threshold, maximum_variance) = (i as u8, variance); }
    }
    let image = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFF } else { 0 }));

    let mut target = Image::zero(image.size);
    for y in R..image.size.y-R {
        for x in R..image.size.x-R {
            let r = (||{ for r in 1..R {
                let circle = std::iter::from_generator(|| {
                    for x in x-r..x+r { yield xy{x,y: y-r}; yield xy{x,y: y+r}; }
                    for y in y-r+1..y+r-1 { yield xy{x: x-r, y}; yield xy{x: x-r, y}; }
                });
                for p in circle { if image[p] == 0 { return r } }
            } unreachable!()})();
            target[xy{x,y}] = r as u8;
        }
    }
    let image = target;

    let mut target = Image::zero(image.size);
    {const R : u32 = 2;
    for y in R..image.size.y-R { for x in R..image.size.x-R {
        let center = image[xy{x, y}];
        let mut flat = 0;
        if (||{ for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            if (dx,dy) == (0,0) { continue; }
            if image[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] > center { return false; }
            if image[xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32}] == center { flat += 1; }
        }} true})() { if flat < 12 { target[xy{x,y}] = center; } }
    }}}
    let image = target;

    let mut image = image;
    let mut points = Vec::new();
    {const R : u32 = 4; // Merges close peaks (top left first)
    for y in R..image.size.y-R { for x in R..image.size.x-R {
        let value = image[xy{x,y}];
        if value == 0 { continue; }
        let mut flat = 0;
        let mut sum : uint2 = zero();
        for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
            if image[p] == 0 { continue; }
            flat += 1;
            sum += p;
        }}
        points.push( (vec2::from(int2::from(xy{x,y}+sum))/((1+flat) as f32), value) );
        for dy in -(R as i32)..=R as i32 { for dx in -(R as i32)..=R as i32 {
            let p = xy{x: (x as i32+dx) as u32, y: (y as i32+dy) as u32};
            image[p] = 0; // Merge small flat plateaus as single point
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
    Q.try_into().unwrap()
}
