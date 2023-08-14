#![feature(generators,iter_from_generator,array_methods,slice_flatten,portable_simd,pointer_byte_offsets,new_uninit,generic_arg_infer,array_try_map,array_windows,slice_take,stdsimd)]
#![allow(non_camel_case_types,non_snake_case,unused_imports)]
use {vector::{xy, uint2, size, int2, vec2}, ::image::{Image,bgr8}, ui::time};

pub trait Camera : Sized {
    fn new() -> Self;
    fn next(&mut self) -> Image<Box<[u16]>>;
    fn next_or_saved_or_start(camera: &mut Option<Self>, clone: &mut Option<Image<Box<[u16]>>>, name: &str, size: size) -> Image<Box<[u16]>> {
        if let Some(camera) = camera.as_mut() {let image = camera.next(); *clone = Some(image.clone()); image} 
        else if let Some(image) = raw(name,size) { image }
        else { *camera = Some(Self::new()); let image = camera.as_mut().unwrap().next(); *clone = Some(image.clone()); image}
    }
}
pub struct Unimplemented;
impl Camera for Unimplemented {
    fn new() -> Self { Self }
    fn next(&mut self) -> Image<Box<[u16]>> { unimplemented!() }
}

#[cfg(feature="nir")] mod nir; #[cfg(feature="nir")] use nir::NIR; #[cfg(not(feature="nir"))] type NIR = Unimplemented;
#[cfg(feature="ir")] mod ir; #[cfg(feature="ir")] use ir::IR; #[cfg(not(feature="ir"))] #[allow(unused)] type IR = Unimplemented;

mod checkerboard; use checkerboard::*;
mod matrix; use matrix::*;
mod image; use image::*;
mod bt40; use bt40::*;

struct App {
    nir: Option<NIR>,
    last_frame: [Option<Image<Box<[u16]>>>; 3],
    debug_which: &'static str,
    debug: &'static str,
    last: Option<[vec2; 4]>,
    calibration: Option<[vec2; 4]>,
    _bt40: Option<BT40>,
    _ir: Option<IR>,
    align: bool,
}
impl App { fn new() -> Self { Self{
    nir: Some(NIR::new()), 
    last_frame: [None, None, None], 
    debug_which: "visible", debug:"", 
    last: None, 
    calibration: if true { std::fs::read("calibration").map(|data| *bytemuck::from_bytes(&data)).ok() } else { Some({
        let nir_size = xy{x: 2048, y: 1152};
        let size = xy{x: nir_size.y*7/5/2, y: nir_size.y/2};
        let offset = (nir_size-size)/2;
        let side_length = size.x/7;
        [xy{x: side_length, y: side_length}, xy{x: size.x-side_length, y: side_length}, xy{x: size.x-side_length, y: size.y-side_length}, xy{x: side_length, y: size.y-side_length}].map(|p| vec2::from(offset+p))
    })},
    _bt40: /*{let mut bt40=BT40::new(); bt40.enable(); bt40}*/None, 
    _ir: None,
    align: false,
}}}
impl ui::Widget for App {
    fn size(&mut self, _: size) -> size { xy{x:2592,y:1944} }
    fn paint(&mut self, target: &mut ui::Target, _: ui::size, _: ui::int2) -> ui::Result {
        if target.size == (xy{x: 960, y: 540}) { return Ok(()); } // WORKAROUND: winit first request paint at wrong scale factor
        
        /*let hololens = Camera::next_or_saved_or_start(&mut self.hololens,&mut self.last_frame[0], "hololens",xy{x:1280,y:720});
        let debug = if self.debug_which=="hololens" {self.debug} else{""};
        if debug=="original" { scale(target, hololens.as_ref()); return Ok(()) }
        let points = match checkerboard_direct_intersections(hololens.as_ref(), 64, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(()) }
        let P_hololens = long_edge_first(top_left_first(simplify(convex_hull(&points.into_iter().map(vec2::from).collect::<Box<_>>())).try_into().unwrap()));
        if debug=="quads" { let (_, scale, offset) = scale(target, hololens.as_ref()); for a in P_hololens { cross(target, scale, offset, a, 0xFF00FF); } return Ok(()) }*/

        let full_nir = Camera::next_or_saved_or_start(&mut self.nir, &mut self.last_frame[1], "nir",xy{x:2592,y:1944});
        //let nir = {let size = nir.size/128*128; nir.slice((nir.size-size)/2, size)}; // 2592x1944 => 2560x1920
        //let nir = {let size = xy{x: nir.size.x, y: nir.size.x/16*9}; nir.slice((nir.size-size)/2, size)};  // 16:9 => 2592x1458|2560x1440
        let nir = {let ref nir = full_nir; let size = xy{x: nir.size.x/128/16*16*128, y: nir.size.x/128/16*9*128}; nir.slice((nir.size-size)/2, size)};  // 16:9 => 2048x1152
        //let nir = {let x = nir.size.x/64*64; nir.slice(xy{x: (nir.size.x-x)/2, y: 0}, xy{x, y: nir.size.y})}; // Both dimensions need to be aligned because of transpose (FIXME: only align stride+height)

        let scale = num::Ratio{num: 15, div: 16}; // 2048->1920 x 1152->1080
        assert_eq!(scale*nir.size, target.size);
        let P_nir_identity = {
            let size = uint2::from(0.5*vec2::from(xy{x: target.size.y*7/5, y: target.size.y}));
            let offset = xy{x:(nir.size.x-size.x)/2, y: nir.size.y-size.y}; // Use bottom part
            //let offset = xy{x:(nir.size.x-size.x)/2, y: 0}; // Use top part
            {
                //[xy{x: , y: 0.}, xy{x: nir.size.x as f32, y: 0.}, xy{x: nir.size.x as f32, y: nir.size.y as f32}, xy{x: 0., y: nir.size.y as f32}]
                // Inner corners
                let side_length = size.x/7;
                [xy{x: side_length, y: side_length}, xy{x: size.x-side_length, y: side_length}, xy{x: size.x-side_length, y: size.y-side_length}, xy{x: side_length, y: size.y-side_length}]
            }.map(|p| vec2::from(offset+p))
        };
        
        let map = |P:[[vec2; 4]; 2]| -> mat3 {
            let M = P.map(|P| {
                let center = P.into_iter().sum::<vec2>() / P.len() as f32;
                let scale = P.len() as f32 / P.iter().map(|p| (p-center).map(f32::abs)).sum::<vec2>();
                [[scale.x, 0., -scale.x*center.x], [0., scale.y, -scale.y*center.y], [0., 0., 1.]]
            });
            let A = homography([P[1].map(|p| apply(M[1], p)), P[0].map(|p| apply(M[0], p))]);
            mul(inverse(M[1]), mul(A, M[0]))
        };
        
        if let Some(calibration) = self.calibration {
            {let size=target.size; for x in 0..target.size.x { target[xy{x, y:0}] = 0xFFFFFF; target[xy{x, y: size.y-1}] = 0xFFFFFF; }}
            {let size=target.size; for y in 0..target.size.y { target[xy{x: 0, y}] = 0xFFFFFF; target[xy{x: size.x-1, y}] = 0xFFFFFF; }}
            if self.align {
                let A = map([P_nir_identity, calibration]);
                let vector::MinMax{min, max} = vector::minmax(nir.iter().copied()).unwrap(); // FIXME
                let scale = 1./f32::from(scale);
                //let ref full_nir = nir;
                let source = full_nir.as_ref();
                for y in 0..target.size.y {
                    for x in 0..target.size.x {
                        let p = scale*xy{x: x as f32, y: y as f32}; // ->nir
                        let p = apply(A, p);
                        let offset = (full_nir.size-nir.size)/2;
                        let p = vec2::from(offset)+p;
                        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
                        let s = source[uint2::from(p)];
                        target[xy{x,y}] = u32::from(bgr8::from(((s-min)*0xFF/(max-min)) as u8)); // nearest: 2048x1152 -> 1920x1080
                    }
                }
            } else {
                let vector::MinMax{min, max} = vector::minmax(nir.iter().copied()).unwrap(); // FIXME
                let scale = 1./f32::from(scale);
                let source = full_nir.as_ref();
                for y in 0..target.size.y {
                    for x in 0..target.size.x {
                        let p = scale*xy{x: x as f32, y: y as f32}; // ->nir
                        let offset = (full_nir.size-nir.size)/2;
                        let p = vec2::from(offset)+p;
                        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
                        let s = source[uint2::from(p)];
                        target[xy{x,y}] = u32::from(bgr8::from(((s-min)*0xFF/(max-min)) as u8)); // nearest: 2048x1152 -> 1920x1080
                    }
                }
            }
            return Ok(());
        }
        
        let debug = if self.debug_which=="visible" {self.debug} else{""};
        if debug=="original" { image::scale(target, nir.as_ref()); return Ok(()) }
        let low = blur::<4>(nir.as_ref()); // 9ms
        if debug=="blur" { image::scale(target, low.as_ref()); return Ok(()) }
        /*let high = self::normalize::<42>(low.as_ref(), 0x1000);
        if debug=="normalize" { scale(target, high.as_ref()); return Ok(()) }*/
        let high = low.as_ref();
        //let Some(P_nir) = checkerboard_quad_debug(high.as_ref(), true, 0/*Blur is enough*//*9*/, /*min_side: */64., debug, target)  else { return Ok(()) };*/
        //let high = nir.as_ref();
        let points = match checkerboard_direct_intersections(high.as_ref(), /*max_distance:*/64, debug) { Ok(points) => points, Err(image) => {image::scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = image::scale(target, high.as_ref()); for &a in &points { cross(target, scale, offset, a.into(), 0xFF0000); } return Ok(()) }
        let P_nir = simplify(convex_hull(&points.iter().copied().map(vec2::from).collect::<Box<_>>())).try_into();
        let P_nir = P_nir.map(|P_nir| long_edge_first(top_left_first(P_nir)));
        let Ok(P_nir) = P_nir else { let (_, scale, offset) = image::scale(target, high.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00000); } return Ok(()); };
        self.last = Some(P_nir);
        if debug=="quads" { let (_, scale, offset) = image::scale(target, nir.as_ref()); for a in P_nir { cross(target, scale, offset, a, 0xFF0000); } return Ok(()) }

        /*let ir = Camera::next_or_saved_or_start(&mut self.ir, &mut self.last_frame[2], "ir",xy{x:256,y:192});
        let debug = if self.debug_which=="ir" {self.debug} else{""};
        if debug=="original" { scale(target, ir.as_ref()); return Ok(()); }
        let Some(high) = self::high_pass(ir.as_ref(), 16, 0x2000) else { scale(target, ir.as_ref()); return Ok(()); };
        let points = match checkerboard_direct_intersections(high.as_ref(), 32, debug) { Ok(points) => points, Err(image) => {scale(target, image.as_ref()); return Ok(()) }};
        if debug=="points" { let (_, scale, offset) = scale(target, ir.as_ref()); for a in points { cross(target, scale, offset, a.into(), 0xFF00FF); } return Ok(())}
        let P_ir = fit_rectangle(&points);
        if debug=="quads" { let (_, scale, offset) = scale(target, ir.as_ref()); for a in P_ir { cross(target, scale, offset, a, 0xFF00FF); } return Ok(())}*/
        
        /*let (target_size, scale, offset) = scale(target, hololens.as_ref());
        for (i,&p) in P_hololens.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }    
        affine_blit(target, target_size, nir.as_ref(), map([P_hololens, P_nir]), hololens.size, 2);
        affine_blit(target, target_size, ir.as_ref(), map([P_hololens, P_ir]), hololens.size, 1);*/
        if let Some(calibration) = self.calibration {
            {let size=target.size; for x in 0..target.size.x { /*target[xy{x, y:0}] = 0xFFFFFF;*/ target[xy{x, y: size.y-1}] = 0xFFFFFF; }}
            //let (_target_size, scale, offset) = scale(target, nir.as_ref());
            {
                let A = map([P_nir_identity, calibration]);
                let vector::MinMax{min, max} = vector::minmax(nir.iter().copied()).unwrap(); // FIXME
                let scale = 1./f32::from(scale);
                let source = nir.as_ref();
                for y in 0..target.size.y {
                    for x in 0..target.size.x {
                        let p = scale*xy{x: x as f32, y: y as f32}; // ->nir
                        let p = apply(A, p);
                        if p.x < 0. || p.x >= source.size.x as f32 || p.y < 0. || p.y >= source.size.y as f32 { continue; }
                        let s = source[uint2::from(p)];
                        target[xy{x,y}] = u32::from(bgr8::from(((s-min)*0xFF/(max-min)) as u8)); // nearest: 2048x1152 -> 1920x1080
                    }
                }
            }
            let offset = xy{x: 0, y: 0};
            let A = map([calibration, P_nir_identity]);
            let P = P_nir.map(|p| apply(A, p));
            //assert_eq!(calibration.map(|p| apply(map, p)), P_nir_identity);
            let scale = f32::from(scale);
            for (i,&p) in P.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); }
            let [p00,p10,p11,p01] = P;
            for i in 0..3 { for j in 0..5 {
                let color = if (i%2==1) ^ (j%2==1) { 0xFFFFFF } else { 0 };
                let p = |di,dj| {
                    let [i, j] = [i+di, j+dj];
                    let [u, v] = [i as f32/3., j as f32/5.];
                    let p0 = (1.-u)*p00+u*p01;
                    let p1 = (1.-u)*p10+u*p11;
                    let p = (1.-v)*p0+v*p1;
                    vec2::from(offset)+scale*p
                };
                let P : [vec2; 4] = [p(0,0),p(0,1),p(1,1),p(1,0)];
                let vector::MinMax{min, max} = vector::minmax(P).unwrap();
                let [p00,p10,p11,p01] = P;
                for y in min.y as u32..=max.y as u32 { for x in min.x as u32..=max.x as u32 {
                    let p = vec2::from(xy{x,y});
                    if [p00,p10,p11,p01,p00].array_windows().all(|&[a,b]| vector::cross2(p-a,b-a)<0.) { if let Some(target) = target.get_mut(xy{x,y}) { *target |= color; } }
                }}
            }}
        } else {
            {let size=target.size; for x in 0..target.size.x { target[xy{x, y:0}] = 0xFFFFFF; target[xy{x, y: size.y-1}] = 0xFFFFFF; }}
            //{let (_, scale, offset) = image::scale(target, nir.as_ref()); for a in P_nir { cross(target, scale, offset, a, 0x00FF00); }} // DEBUG
            {
                let size = uint2::from(0.5*vec2::from(xy{x: target.size.y*7/5, y: target.size.y}));
                let offset = xy{x:(target.size.x-size.x)/2, y: target.size.y-size.y}; // Use bottom part
                //let offset = xy{x:(target.size.x-size.x)/2, y: 0}; // Use top part
                let mut target = target.slice_mut(offset, size);
                let side_length = target.size.x/7;
                for y in 0..size.y {
                    for x in 0..size.x {
                        let [j, i] = [x,y].map(|x| x/side_length);
                        target[xy{x,y}] |= if (i%2==1) ^ (j%2==1) { 0xFFFFFF } else { 0 };
                    }
                }
            }
            /*{
                let offset = xy{x: 0, y: 0};
                let scale = f32::from(scale);
                for (i,&p) in P_nir_identity.iter().enumerate() { cross(target, scale, offset, p, [0xFF_0000,0x00_FF00,0x00_00FF,0xFF_FFFF][i]); } //DEBUG
            }*/
        }
        Ok(())
    }
    fn event(&mut self, _: vector::size, context: &mut ui::EventContext, event: &ui::Event) -> ui::Result<bool> {
        if let &ui::Event::Key(key) = event {
            if ['⎙','\u{F70C}'].contains(&key) {
                println!("⎙");
                write("checkerboard.png", {let mut target = Image::uninitialized(xy{x: 2592, y:1944}); let size = target.size; self.paint(&mut target.as_mut(), size, xy{x: 0, y: 0}).unwrap(); target}.as_ref());
                if self.last_frame.iter_mut().zip([/*"hololens",*/"nir"/*,"ir"*/]).filter_map(|(o,name)| o.take().map(|o| (name, o))).inspect(|(name, image)| write_raw(name, image.as_ref())).count() == 0 {
                    //if self.hololens.is_none() { self.hololens = Some(Hololens::new()); }
                    if self.nir.is_none() { self.nir = Some(NIR::new()); }
                    //if self.ir.is_none() { self.ir = Some(IR::new()); }
                }
            } else if key=='\n' {
                if self.calibration.is_none() {
                    self.calibration = self.last;
                    //context.set_title(&format!("{}", self.calibration.is_some()));
                    std::fs::write("calibration", bytemuck::bytes_of(&self.calibration.unwrap())).unwrap();
                } else {
                    self.calibration = None;
                }
            } else {
                fn starts_with<'t>(words: &[&'t str], key: char) -> Option<&'t str> { words.into_iter().find(|word| key==word.chars().next().unwrap()).copied() }
                if let Some(word) = starts_with(&[/*"hololens",*/"visible"/*,"ir"*/], key) { self.debug_which = word }
                else if let Some(word) = starts_with(&["blur","contour","response","erode","normalize","low","max","original","z","points","quads","threshold"], key) { self.debug = word; }
                //else { self.debug_which=""; self.debug="";  }
                else { self.align = !self.align; }
                context.set_title(&format!("{} {}", self.debug_which, self.debug));
            }
            Ok(true)
        } else { Ok(/*self.hololens.is_some() ||*/ self.nir.is_some() /*||self.ir.is_some()*/) }
    }
}
#[cfg(debug_assertions)] const TITLE: &'static str = "#[cfg(debug_assertions)]";
#[cfg(not(debug_assertions))] const TITLE: &'static str = "Checkerboard";
fn main() -> ui::Result { ui::run(TITLE, &mut App::new()) }
