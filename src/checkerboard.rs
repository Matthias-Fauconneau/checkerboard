use {num::{sq, zero}, vector::{xy, uint2, int2, vec2, cross2, norm, minmax, MinMax}, image::Image};

pub fn transpose_box_convolve_scale<const R: usize>(source: Image<&[u16]>, factor: u32) -> Image<Box<[u16]>> {
    let mut transpose = Image::uninitialized(source.size.yx());
    assert!(R+1+R < (1<<4) && R+1+R > (1<<(4-1)));
    let stride = source.stride as usize;
    let height = source.size.y as usize;
    //let stride : [_; R] = std::array::from_fn(|y| stride as usize*y);
    let mut column = source.data.as_ptr();
    let mut row : *mut u16 = transpose.data.as_mut_ptr();
    assert!(transpose.stride as usize == height);
    for _x in 0..source.size.x { unsafe {
        let first = *column as u32;
        let mut sum = first*(R as u32);
        let mut front = column;
        for _y in 0..R { sum += *front as u32; front = front.add(stride); }
        let end = row.add(height).sub(R+1);
        for _y in 0..R {
            sum += *front as u32;
            front = front.add(stride);
            *row = (sum * factor >> (16-4)) as u16;
            sum -= first;
            row = row.add(1);
        }
        let mut back = column;
        while row < end { //for y in R..height-R-1*/
            sum += *front as u32;
            front = front.add(stride);
            *row = (sum * factor >> (16-4)) as u16;
            row = row.add(1);
            sum -= *back as u32;
            back = back.add(stride);
        }
        // height-R-1
        let last = *front as u32;
        let end = end.add(R+1);
        while row < end { // for y in height-R-1..height
            sum += last;
            *row = (sum * factor >> (16-4)) as u16;
            row = row.add(1);
            sum -= *back as u32;
            back = back.add(stride);
        }
        column = column.add(1);
    }}
    transpose
}

pub fn transpose_box_convolve<const R: usize>(source: Image<&[u16]>) -> Image<Box<[u16]>> { transpose_box_convolve_scale::<R>(source, (1<<(16-4)) / (R+1+R) as u32) }

pub fn transpose_box_convolve_scale_minmax<const R: usize>(source: Image<&[u16]>) -> Image<Box<[u16]>> {
    let mut max = 0;
    let std::ops::Range{start: mut sample, end} = source.data.as_ptr_range();
    unsafe{while sample < end { max = u16::max(max, *sample); sample = sample.add(1); }}
    transpose_box_convolve_scale::<R>(source, (1<<(32-4)) / (max as u32*(R+1+R) as u32))
}

pub fn box_convolve<const R: usize>(image: Image<&[u16]>) -> Image<Box<[u16]>> { transpose_box_convolve::<R>(transpose_box_convolve_scale_minmax::<R>(image).as_ref()) }

/*pub fn high_pass(image: Image<&[u16]>, R: u32, threshold: u16) -> Option<Image<Box<[u16]>>> {
    let vector::MinMax{min, max} = vector::minmax(image.iter().copied()).unwrap();
    if !(min < max) { return None; }
    let mut low_then_high = box_convolve(image.as_ref(), R);
    low_then_high.as_mut().zip_map(&image, |&low, &p| {
        let high = (p-min) as u32*0xFFFF/(max-min) as u32;
        assert!(high <= 0xFFFF);
        let low = low.max(threshold);
        let div = ((high*0xFFFF/low as u32)>>3) as u16;
        div
    });
    Some(low_then_high)
}*/

fn otsu(image: Image<&[u16]>) -> u16 {
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
    threshold
}

fn binary16(image: Image<&[u16]>, threshold: u16, inverse: bool) -> Image<Box<[u16]>> { Image::from_iter(image.size, image.iter().map(|&p| match inverse {
    false => if p>threshold { 0xFFFF } else { 0 },
    true => if p<threshold { 0xFFFF } else { 0 }
}))}
fn binary(image: Image<&[u16]>, threshold: u16, inverse: bool) -> Image<Box<[u8]>> { Image::from_iter(image.size, image.iter().map(|&p| match inverse {
    false => if p>threshold { 0xFF } else { 0 },
    true => if p<threshold { 0xFF } else { 0 }
}))}

fn erode(high: Image<&[u16]>, erode_steps: usize, threshold: u16) -> Image<Box<[u8]>> {
    let mut binary = self::binary(high.as_ref(), threshold, false);
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
        if [a,b,c,d,a].array_windows().any(|&[a,b]| vector::sq(b-a)>256.*256.) {continue;}
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

pub fn simplify(mut P: Vec<vec2>) -> Vec<vec2> {
    // Simplifies polygon to 4 corners
    while P.len() > 4 {
        P.remove(((0..P.len()).map(|i| {
            let [p0, p1, p2]  = std::array::from_fn(|j| P[(i+j)%P.len()]);
            (cross2(p2-p0, p1-p0), i)
        }).min_by(|(a,_),(b,_)| a.total_cmp(b)).unwrap().1+1)%P.len());
    }
    P
}

pub fn top_left_first(Q: [vec2; 4]) -> [vec2; 4] { let i0 = Q.iter().enumerate().min_by(|(_,a),(_,b)| (a.x+a.y).total_cmp(&(b.x+b.y))).unwrap().0; [0,1,2,3].map(|i|Q[(i0+i)%4]) }
pub fn long_edge_first(mut Q: [vec2; 4]) -> [vec2; 4] { if norm(Q[2]-Q[1])+norm(Q[0]-Q[3]) > norm(Q[1]-Q[0])+norm(Q[3]-Q[2]) { Q.swap(1,3); } Q }

pub fn checkerboard_quad(high: Image<&[u16]>, _black: bool, erode_steps: usize, min_side: f32, debug: &'static str) -> Result<[vec2; 4],Image<Box<[u16]>>> {
    let threshold = otsu(high.as_ref());    
    if debug=="binary" { return Err(binary16(high.as_ref(), threshold, false)); }
    let erode = erode(high.as_ref(), erode_steps, threshold);
    if debug=="erode" { return Err(Image::from_iter(erode.size, erode.iter().map(|&p| (p as u16)<<8))); }
    let ref contours = imageproc::contours::find_contours::<u16>(&imagers::GrayImage::from_vec(erode.size.x, erode.size.y, erode.data.into_vec()).unwrap());
    if debug=="contour" { let mut target = Image::zero(erode.size); for contour in contours { for p in &contour.points { target[xy{x: p.x as u32, y: p.y as u32}] = 0xFFFF; }} return Err(target); }
    let quads = quads(contours, sq(min_side));
    if debug=="quads" { let mut target = Image::zero(erode.size);
        for q in &*quads { for [&a,&b] in {let [a,b,c,d] = q; [a,b,c,d,a]}.array_windows() { for (p,_,_,_) in ui::line::generate_line(target.size, [a,b]) { target[p] = 0xFFFF; } } }
        return Err(target); 
    }
    let points : Box<_> = quads.into_iter().map(|q| q.into_iter()).flatten().copied().collect();
    if points.len() < 4 { return Err(binary16(high.as_ref(), threshold, false)); }
    let Q = simplify(convex_hull(&*points));
    let Ok(Q) = Q.try_into() else { return Err(binary16(high, threshold, false)) };
    Ok(long_edge_first(top_left_first(Q)))
}

/*fn cross_response(high: Image<&[u16]>) -> Image<Box<[u16]>> {
    let mut cross = Image::zero(high.size);
    const R : u32 = 3;
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
    low_pass(cross.as_ref(), 1)
}

pub fn checkerboard_direct_intersections(high: Image<&[u16]>, max_distance: u32, debug: &'static str) -> std::result::Result<Vec<uint2>,Image<Box<[u16]>>> {
    let cross = self::cross_response(high);
    if debug=="response" { return Err(cross); }
    
    let threshold = otsu(cross.as_ref()); //foreground_mean
    if debug=="binary" { return Err(Image::from_iter(cross.size, cross.iter().map(|&p| if p>threshold { p } else { 0 }))); }

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
    let points : Vec<_> = points.iter().map(|&a| (a, {
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
    }).max_by_key(|g| g.len()).unwrap())
}

pub fn refine(high: Image<&[u16]>, mut points: [vec2; 4], R: u32, debug: &'static str) -> Result<[vec2; 4],(vec2, Box<[vec2]>, Box<[vec2]>, Image<Box<[uint2]>>, vec2, vec2, Image<Box<[u16]>>)> {
    let cross = cross_response(high.as_ref());

    for _ in 0..1 {
        let grid = {
            let cross = &cross;
            Image::from_iter(xy{x:8,y:6},
          (1..=6).map(|y|
                    (1..=8).map(move |x| {
                        let xy{x, y} = xy{x: x as f32/9., y: y as f32/7.};
                        let [A,B,C,D] = points.clone();
                        let p = y*(x*A+(1.-x)*B) + (1.-y)*(x*D+(1.-x)*C);
                        let p = xy{x: p.x.round() as u32, y: p.y.round() as u32,};
                        let (p,_) = (p.y-R..p.y+R).map(|y| (p.x-R..p.x+R).map(move |x| xy{x,y})).flatten().map(|p| (p, cross[p])).max_by_key(|(_,v)| *v).unwrap_or((p,0));
                        p
                    }
                    )
                ).flatten()
            )
        };
        let ref ref_grid = grid;
        let rows = (0..grid.size.y).map(|y| (0..grid.size.x).map(move |x| ref_grid[xy{x,y}]));
        let columns = (0..grid.size.x).map(|x| (0..grid.size.y).map(move|y| ref_grid[xy{x,y}]));
        let column = rows.map(|row| row.map(|p| vec2::from(p)).sum::<vec2>()/8.).collect::<Box<_>>();
        let row = columns.map(|column| column.map(|p| vec2::from(p)).sum::<vec2>()/6.).collect::<Box<_>>();
        let row_axis = row.last().unwrap()-row[0];
        let column_axis = column.last().unwrap()-column[0];
        let center = grid.iter().map(|&p| vec2::from(p)).sum::<vec2>()/(8.*6.);
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

pub fn checkerboard_quad_debug(nir: Image<&[u16]>, black: bool, erode_steps: usize, min_side: f32, debug: &'static str, target: &mut Image<&mut [u32]>) -> Option<[vec2; 4]> { 
    use super::image::scale;
    match checkerboard_quad(nir.as_ref(), black, erode_steps, min_side, debug) {
        Err(image) => { scale(target, image.as_ref()); None }
        Ok(points) => {
            let points = match refine(nir.as_ref(), points, 0/*128*/, debug) {
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
            };
            if debug=="z" {
                let (_, scale, offset) = scale(target, nir.as_ref());
                for (i,&p) in points.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
                return None;
            }
            Some(points)
        }
    }
}*/

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