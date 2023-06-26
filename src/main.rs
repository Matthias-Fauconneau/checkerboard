#![feature(generators,iter_from_generator)]#![allow(non_camel_case_types)]
pub type Error = Box<dyn std::error::Error>; use {std::cmp::{min,max}, num::{sq,sqrt}, fehler::throws, vector::{xy, int2, size}, image::Image, ui::{Widget, Target}};
fn main() {
    let image = imagers::open("ir.png").unwrap();
    
    /*use checkerboard::{CheckerboardSpecification, rufli::detect_checkerboard};
    let quads = detect_checkerboard(&image, &CheckerboardSpecification{width: 9, height: 7}).unwrap();
    println!("{quads:?}");*/

    let image = Image::new(vector::xy{x: image.width(), y: image.height()}, image.as_bytes());

    /*const R : u32 = 8;
    let mut target = Image::zero(image.size);
    for y in R..image.size.y-R {
        for x in R..image.size.x-R {
            let mut sum = image[xy{x,y}] as u16;
            let mut count = 1;
            let r = (||{ for r in 1.. {
                let mean = (sum / count) as u8;
                /*let f = |x,y| {

                    sum += image[xy{x,y: y-r}];
                }
                for x in x-r..x+r { sum += image[xy{x,y: y-r}]; sum += image[xy{x,y: y+r}]; }
                for y in y-r+1..y+r-1 { sum += image[xy{x: x-r, y}]; sum += image[xy{x: x-r, y}]; }*/
                let circle = std::iter::from_generator(|| {
                    for x in x-r..x+r { yield xy{x,y: y-r}; yield xy{x,y: y+r}; }
                    for y in y-r+1..y+r-1 { yield xy{x: x-r, y}; yield xy{x: x-r, y}; }
                });
                for p in circle {
                    if image[p] < mean { return r }
                    sum += image[p] as u16;
                }
                count += (r+1+r+(r-1)+1+(r-1)) as u16;
            } unreachable!()})();
            target[xy{x,y}] = r as u8;
        }
    }
    let max = *target.iter().max().unwrap();
    target.as_mut().map(|&v| (v as u16 * 0xFF / max as u16) as u8);
    let image = target;*/

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
        //let foreground_mean = foreground_sum/foreground_count;
        //let background_mean = background_sum/background_count;
        type u40 = u64;
        /*assert!(foreground_sum < 1<<24);
        assert!(background_sum < 1<<24);*/
        let variance = sq((foreground_sum as u40*background_count as u40 - background_sum as u40*foreground_count as u40) as f64)/(foreground_count as u32*background_count as u32) as f64;
        if variance >= maximum_variance { (threshold, maximum_variance) = (i as u8, variance); }
    }
    let image = Image::from_iter(image.size, image.iter().map(|&p| if p>threshold { 0xFF } else { 0 }));

    /*// Erosion
    let mut target = Image::zero(image.size);
    for y in 0..image.size.y { for x in 0..image.size.x {
        let mut min = 0xFF;
        for y in max(0, y as i32-1) as u32 .. self::min(image.size.y-1, y+1) { for x in max(0, x as i32-1) as u32 .. self::min(image.size.x-1, x+1) {
            min = self::min(min, image[xy{x,y}]);
        }}
        target[xy{x,y}] = min;
    }}
    let image = target;*/

    //const R : u32 = 8;
    let mut target = Image::zero(image.size);
    for y in R..image.size.y-R {
        for x in R..image.size.x-R {
            let r = (||{ for r in 1.. {
                let circle = std::iter::from_generator(|| {
                    for x in x-r..x+r { yield xy{x,y: y-r}; yield xy{x,y: y+r}; }
                    for y in y-r+1..y+r-1 { yield xy{x: x-r, y}; yield xy{x: x-r, y}; }
                });
                for p in circle { if image[p] == 0 { return r } }
            } unreachable!()})();
            target[xy{x,y}] = r as u8;
        }
    }
    let max = *target.iter().max().unwrap();
    target.as_mut().map(|&v| (v as u16 * 0xFF / max as u16) as u8);
    let image = target;

    struct View<'t>(Image<&'t [u8]>);
    impl Widget for View<'_> {
        #[throws] fn paint(&mut self, target: &mut Target, _: size, _: int2) {
            let ref source = self.0;
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
    }
    ui::run("Checkerboard", &mut View(image.as_ref())).unwrap();
}
