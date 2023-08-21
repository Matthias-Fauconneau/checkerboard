//sudo sh -c 'echo 32 > /sys/module/usbcore/parameters/usbfs_memory_mb'
#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows,slice_take,stdsimd,iter_map_windows)]
#![allow(non_camel_case_types,non_snake_case,unused_imports,dead_code)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}, ui::time};

pub trait Camera : Sized {
    fn new(serial_number: &str) -> Self;
    fn next(&mut self) -> Image<Box<[u16]>>;
}
mod nir; use nir::NIR;
mod ir; use ir::IR; 

mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;
mod bt40; use bt40::*;

#[derive(Debug,Clone,Copy)] enum Mode { Place, Calibrate, Use, Replace }
struct Calibrated<T> {
    camera: T,
    calibration: [[vec2; 4]; 2],
}
impl<T:Camera> Calibrated<T> {
    const IDENTITY : [vec2; 4] = [xy{x: 0., y: 0.}, xy{x: 1., y: 0.}, xy{x: 1., y: 1.}, xy{x: 0., y: 1.}];
    fn new(name: &str, serial_number: &str, rotate: bool) -> Self { 
        Self{
            camera: T::new(serial_number), 
            calibration: std::fs::read(name).ok().filter(|data| data.len()==std::mem::size_of::<[[vec2; 4]; 2]>()).map(|data| *bytemuck::from_bytes(&data)).unwrap_or(
                if rotate { [Self::IDENTITY, [xy{x: 1., y: 1.},xy{x: 0., y: 1.}, xy{x: 0., y: 0.}, xy{x: 1., y: 0.}]] }
                else { [Self::IDENTITY; 2] }
            )
        }
    }
}
struct App {
    nir: Calibrated<NIR>,
    ir: Calibrated<IR>,
    fluo: Calibrated<NIR>,
    mode: Mode,
    last: Option<[vec2; 4]>,
    _bt40: Option<BT40>,
    which: &'static str,
    debug: &'static str,
    //last_frame: [Option<Image<Box<[u16]>>>; 3],
    save: bool,
    min: uint2, max: uint2 // Fast clear
}
impl App { 
    fn new() -> Self { Self{
        nir: Calibrated::<NIR>::new("nir","4104585450", false),
        ir: Calibrated::<IR>::new("ir","", false),
        fluo: Calibrated::<NIR>::new("fluo","4104590291", true),
        mode: Mode::Use,
        last: None, 
        _bt40: /*{let mut bt40=BT40::new(); bt40.enable(); bt40}*/None, 
        which: "fluo", debug:"", 
        //last_frame: [None, None, None], 
        save: false,
        min: xy{x: 0, y: 0}, max: xy{x: 0, y: 0},
    }}
    fn calibration(&mut self) -> &mut [[vec2; 4]; 2] {
        match self.which {
            "nir" => &mut self.nir.calibration,
            "ir" => &mut self.ir.calibration,
            "fluo" => &mut self.fluo.calibration,
            _ => unimplemented!(),
        }
    }
}
impl ui::Widget for App {
    fn size(&mut self, _: size) -> size { xy{x:2592,y:1944} }
    fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
        if target.size == (xy{x: 960, y: 540}) { return Ok(()); } // WORKAROUND: winit first request paint at wrong scale factor
        
        let full_nir = self.nir.camera.next();
        let nir = {let ref nir = full_nir; let size = xy{x: nir.size.x/128/16*16*128, y: nir.size.x/128/16*9*128}; nir.slice((nir.size-size)/2, size)};  // 16:9 => 2048x1152
        
        let ir = self.ir.camera.next();

        let full_fluo = self.fluo.camera.next();
        let fluo = {let ref fluo = full_fluo; let size = xy{x: fluo.size.x/128/16*16*128, y: fluo.size.x/128/16*9*128}; fluo.slice((fluo.size-size)/2, size)};  // 16:9 => 2048x1152

        if let Mode::Use = self.mode {
            let ref ir_nir_fluo = [(ir.as_ref(), self.ir.calibration, xy{x: 0, y: 0}), (full_nir.as_ref(), self.nir.calibration, (full_nir.size-nir.size)/2), (full_fluo.as_ref(), self.fluo.calibration, (full_fluo.size-fluo.size)/2)]
            .map(|(source, calibration, source_offset)| {
                let scale = num::Ratio{num: target.size.y, div: source.size.y};
                let target_offset = xy{x: (target.size.x-scale*source.size.x)/2, y: (target.size.y-scale*source.size.y)/2};
                let A = homography(calibration);
                let vector::MinMax{min, max} = vector::minmax(source.iter().copied()).unwrap(); // FIXME
                (source, source_offset, scale, target_offset, A, min, max)
            });
            for y in 0..target.size.y {
                for x in 0..target.size.x {
                    let [ir, nir, fluo] = ir_nir_fluo.each_ref().map(|(source, source_offset, scale, target_offset, A, min, max)| {
                        if !(min<max) { 0 } else {
                            let scale_from_target_to_source = 1./f32::from(*scale);
                            let p = scale_from_target_to_source*(xy{x: x as f32, y: y as f32} - vec2::from(*target_offset)); // target->source
                            let p = apply(*A, p);
                            let p = vec2::from(*source_offset)+p;
                            if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { 0 } else {
                                let s = source[uint2::from(p)];
                                ((s-min)*0xFF/(max-min)) as u8
                            }
                        }
                    });
                    target[xy{x,y}] = u32::from(bgr8::from([ir,nir,fluo]));
                }
            }
            return Ok(());
        }
        let source = match self.which {
            "ir" => ir.as_ref(),
            "nir" => full_nir.as_ref(),
            "fluo" => full_fluo.as_ref(),
            _ => unreachable!()
        };
        let scale = num::Ratio{num: target.size.y, div: source.size.y};
        let target_offset = xy{x: (target.size.x-scale*source.size.x)/2, y: (target.size.y-scale*source.size.y)/2};

        let debug = self.debug;
        let P = match self.which {
            "nir" => {
                if debug=="original" { image::scale(target, nir.as_ref()); return Ok(()) }
                let low = blur::<4, 4>(nir.as_ref()).unwrap(); // 9ms
                if debug=="blur" { image::scale(target, low.as_ref()); return Ok(()) }
                /*let high = self::normalize::<42>(low.as_ref(), 0x1000);
                if debug=="high" { scale(target, high.as_ref()); return Ok(()) }*/
                let high = low.as_ref();
                //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 0/*Blur is enough*/ /*9*/, /*min_side: */64., debug, target)  else { return Ok(()) };
                //let high = nir.as_ref();
                let points = match checkerboard_direct_intersections::<32,4>(high.as_ref(), /*max_distance:*/64, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};
                let Q = convex_hull(&points.iter().copied().map(vec2::from).collect::<Box<_>>());
                if debug=="points" || Q.len() < 4 { let (_, scale, offset) = image::scale(target, high.as_ref()); for &a in &points { cross(target, scale, offset, a.into(), 0xFF0000); } return Ok(()) }
                let P = long_edge_first(top_left_first(simplify(Q)));
                if debug=="quads" { let (_, scale, offset) = image::scale(target, nir.as_ref()); for a in P { cross(target, scale, offset, a, 0xFF0000); } return Ok(()) }
                P
            },
            "fluo" => {
                if debug=="original" { image::scale(target, fluo.as_ref()); return Ok(()) }
                let low = blur::<4, 4>(fluo.as_ref()).unwrap(); // 9ms
                if debug=="blur" { image::scale(target, low.as_ref()); return Ok(()) }
                /*let high = self::normalize::<42>(low.as_ref(), 0x1000);
                if debug=="high" { scale(target, high.as_ref()); return Ok(()) }*/
                let high = low.as_ref();
                //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 0/*Blur is enough*/ /*9*/, /*min_side: */64., debug, target)  else { return Ok(()) };
                //let high = nir.as_ref();
                let points = match checkerboard_direct_intersections::<32,4>(high.as_ref(), /*max_distance:*/64, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};
                let Q = convex_hull(&points.iter().copied().map(vec2::from).collect::<Box<_>>());
                if debug=="points" || Q.len() < 4 { let (_, scale, offset) = image::scale(target, high.as_ref()); for &a in &points { cross(target, scale, offset, a.into(), 0xFF0000); } return Ok(()) }
                let P = long_edge_first(top_left_first(simplify(Q)));
                if debug=="quads" { let (_, scale, offset) = image::scale(target, nir.as_ref()); for a in P { cross(target, scale, offset, a, 0xFF0000); } return Ok(()) }
                P
            }
            "ir" => {
                if debug=="original" { image::scale(target, ir.as_ref()); return Ok(()); }
                /*let Some(low) = blur::<1,2>(ir.as_ref()) else {image::scale(target, ir.as_ref()); return Ok(()) };
                let low = blur_slow(ir.as_ref(), 4);
                if debug=="blur" { image::scale(target, low.as_ref()); return Ok(()) };*/
                let Some(high) = checkerboard::normalize_slow/*::<16, 2>*/(ir.as_ref(), 32/*16*/, 0x1000) else {image::scale(target, ir.as_ref()); return Ok(()) };
                if debug=="high" { image::scale(target, high.as_ref()); return Ok(()) };
                let cross = cross_response::<3, 2>(high.as_ref()).unwrap();
                /*if debug=="z" {
                    let cross = cross_response::<3, 2>(high.as_ref()).unwrap();
                    image::scale(target, cross.as_ref()); return Ok(()) 
                }
                let points = match checkerboard_direct_intersections::<3, 2>(high.as_ref(), 32, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};*/
                if debug=="response" {image::scale(target, cross.as_ref()); return Ok(()) }
                let threshold = checkerboard::otsu(cross.as_ref()); //foreground_mean
                if debug=="threshold" { image::scale(target, Image::from_iter(cross.size, cross.iter().map(|&p| if p>threshold { p } else { 0 })).as_ref()); return Ok(()) }
                let mut points = Vec::new();
                const R : u32 = 6;
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
                if points.len() < 4  { let (_, scale, offset) = image::scale(target, ir.as_ref()); for a in points { checkerboard::cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(())}
                let P = convex_hull(&points.iter().copied().map(vec2::from).collect::<Box<_>>());
                if P.len() < 4 || debug=="points" { let (_, scale, offset) = image::scale(target, ir.as_ref()); for a in points { checkerboard::cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(())}
                let P = simplify(P);
                if debug=="quads" { let (_, scale, offset) = image::scale(target, ir.as_ref()); for a in P { checkerboard::cross(target, scale, offset, a, 0xFF00FF); } return Ok(())}
                P
            },
            _ => unreachable!()
        };

        self.last = Some(P);
        if let Mode::Replace = self.mode {
            self.calibration()[0] = P;
            self.mode = Mode::Calibrate; 
        }

        let P = match self.mode {
            Mode::Place => P,
            Mode::Calibrate => {
                let scale = f32::from(scale);
                for (i,&p) in P.iter().enumerate() { cross(target, scale, target_offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }
                self.calibration()[0]
            },
            mode => unreachable!("{mode:?}"),
        };
        let scale = f32::from(scale);
        let [p00,p10,p11,p01] = P;
        use vector::ComponentWiseMinMax;
        self.min = self.min.component_wise_max(xy{x: 0, y: 0});
        self.max = self.max.component_wise_min(target.size);
        if self.min < self.max { ::image::fill(&mut target.slice_mut(self.min, self.max-self.min), 0); }
        [self.min, self.max] = [target.size, xy{x: 0, y: 0}];
        for i in 0..3 { for j in 0..5 {
            let color = if (i%2==1) ^ (j%2==1) { 0xFFFFFF } else { 0 };
            if color == 0 { continue; }
            let p = |di,dj| {
                let [i, j] = [i+di, j+dj];
                let [u, v] = [i as f32/3., j as f32/5.];
                let p0 = (1.-u)*p00+u*p01;
                let p1 = (1.-u)*p10+u*p11;
                let p = (1.-v)*p0+v*p1;
                vec2::from(target_offset)+scale*p
            };
            let P : [vec2; 4] = [p(0,0),p(0,1),p(1,1),p(1,0)];
            let vector::MinMax{min, max} = vector::minmax(P).unwrap();
            let [p00,p10,p11,p01] = P;
            for y in min.y as u32..=max.y as u32 { for x in min.x as u32..=max.x as u32 {
                let p = vec2::from(xy{x,y});
                if [p00,p10,p11,p01,p00].array_windows().all(|&[a,b]| vector::cross2(p-a,b-a)<0.) { if let Some(target) = target.get_mut(xy{x,y}) { *target = color; } }
            }}
            self.min = self.min.component_wise_min(min.into());
            self.max = self.max.component_wise_max(max.into()); // FIXME ceil
        }}
        Ok(())
    }
    fn event(&mut self, _: vector::size, _context: &mut ui::EventContext, event: &ui::Event) -> ui::Result<bool> {
        if let &ui::Event::Key(key) = event {
            if ['⎙','\u{F70C}'].contains(&key) {
                //println!("⎙");
                //write("checkerboard.png", {let mut target = Image::uninitialized(xy{x: 2592, y:1944}); let size = target.size; self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap(); target}.as_ref());
                /*if self.last_frame.iter_mut().zip([/*"hololens",*/"nir","ir"]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| write_raw(name, image.as_ref())).count() == 0 {
                    //if self.hololens.is_none() { self.hololens = Some(Hololens::new()); }
                    //if self.nir.is_none() { self.nir = Some(Calibrated::<NIR>::new("nir")); }
                    //if self.ir.is_none() { self.ir = Some(Calibrated::<IR>::new("ir")); }
                }*/
                self.save = true;
            } else if key=='\n' {
                if let Some(last)= self.last {
                    self.calibration()[1] = last;
                    //context.set_title(&format!("{}", self.calibration.is_some()));
                    std::fs::write(self.which, bytemuck::bytes_of(&*self.calibration())).unwrap();
                    self.mode = Mode::Use; 
                }
            }
            else if key==' ' { self.mode = Mode::Replace; }
            else if key=='⌫' { 
                *self.calibration() = [Calibrated::<NIR>::IDENTITY; 2];
                if self.which == "fluo" { self.fluo.calibration = [Calibrated::<NIR>::IDENTITY, [xy{x: 1., y: 1.},xy{x: 0., y: 1.}, xy{x: 0., y: 0.}, xy{x: 1., y: 0.}]]; }
                //self.mode = Mode::Place; 
            } 
            else if ['-','+','↑','↓','←','→'].contains(&key) {
                //println!("{:?} {:?}", self.ir.calibration, apply(homography(self.ir.calibration), xy{x:1./2., y:1./2.}));
                let offset = if self.which == "fluo" {32.} else if self.which == "nir" { 8. } else { 1. };
                self.calibration()[1] = self.calibration()[1].map(|p| match key {
                    '-' => (1.-0.1)*p,
                    '+' => 1.1*p,
                    '↑' => xy{x: p.x, y: p.y-offset},
                    '↓' => xy{x: p.x, y: p.y+offset},
                    '←' => xy{x: p.x-offset, y: p.y},
                    '→' => xy{x: p.x+offset, y: p.y},
                    _ => unreachable!()
                });
                //println!("{:?} {:?}", self.ir.calibration, apply(homography(self.ir.calibration), xy{x:1./2., y:1./2.}));
                std::fs::write(self.which, bytemuck::bytes_of(&*self.calibration())).unwrap();
            } else {
                fn starts_with<'t>(words: &[&'t str], key: char) -> Option<&'t str> { words.into_iter().find(|word| key==word.chars().next().unwrap()).copied() }
                if let Some(word) = starts_with(&["nir","ir","fluo"], key) { self.which = word }
                else if let Some(word) = starts_with(&["blur","contour","response","erode","high","low","max","original","z","points","quads","threshold"], key) { self.debug = word; }
                else { self.debug="";  }
                //context.set_title(&format!("{} {}", self.which, self.debug));
            }
            Ok(true)
        } else { Ok(/*self.hololens.is_some() ||*/ /*self.nir.is_some()*/ /*||self.ir.is_some()*/true) }
    }
}
#[cfg(debug_assertions)] const TITLE: &'static str = "#[cfg(debug_assertions)]";
#[cfg(not(debug_assertions))] const TITLE: &'static str = "Checkerboard";
fn main() -> ui::Result { 
    if false {
        struct Test {}
        impl ui::Widget for Test {
            fn paint(&mut self, _target: &mut ui::Target, _size: size, _offset: int2) -> ui::Result {

                Ok(())
            }
        }
        ui::run(TITLE, &mut Test{}) 
    } else {
        ui::run(TITLE, &mut App::new()) 
    }
}
