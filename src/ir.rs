#[cfg(not(feature="uvc"))] pub struct IR;
#[cfg(feature="uvc")] pub struct IR(*mut uvc::uvc_stream_handle_t);
use image::{Image, xy};
#[cfg(not(feature="uvc"))] impl IR {
    pub fn new() -> Self { Self }
    pub fn next(&mut self) -> Image<Box<[u16]>> { panic!("!uvc") }
}
#[cfg(feature="uvc")] impl IR {
    pub fn new() -> Self {
        use std::ptr::null_mut;
        let mut uvc = null_mut();
        use uvc::*;
        assert!(unsafe{uvc_init(&mut uvc as *mut _, null_mut())} >= 0);
        let mut devices : *mut *mut uvc_device_t = null_mut();
        assert!(unsafe{uvc_find_devices(uvc, &mut devices as *mut _, 0/*xbda*/, 0/*x5840*/, std::ptr::null())} >= 0);
        for device in std::iter::successors(Some(devices), |devices| Some(unsafe{devices.add(1)})) {
            let device = unsafe{*device};
            if device.is_null() { break; }
            let mut device_descriptor : *mut uvc_device_descriptor_t = null_mut();
            assert!(unsafe{uvc_get_device_descriptor(device, &mut device_descriptor as &mut _)} >= 0);
            assert!(!device_descriptor.is_null());
            let mut device_handle = null_mut();
            assert!(unsafe{uvc_open(device, &mut device_handle as *mut _)} >= 0);
            let mut control = unsafe{std::mem::zeroed()};
            if unsafe{uvc_get_stream_ctrl_format_size(device_handle, &mut control as *mut _, uvc_frame_format_UVC_FRAME_FORMAT_ANY, 256, 192, 25)} < 0 { continue; }
            let mut stream = null_mut();
            assert!(unsafe{uvc_stream_open_ctrl(device_handle, &mut stream as *mut _, &mut control as *mut _)} >= 0);
            assert!(unsafe{uvc_stream_start(stream, None, null_mut(), 0)} >= 0);
            return Self(stream);
        }
        panic!();
    }
    pub fn next(&mut self) -> Image<Box<[u16]>> {
        use uvc::*;
        let mut frame : *mut uvc_frame_t = std::ptr::null_mut();
        assert!(unsafe{uvc_stream_get_frame(self.0, &mut frame as *mut _, 1000000)} >= 0);
        assert!(!frame.is_null());
        let uvc_frame_t{width, height, data, data_bytes, ..} = unsafe{*frame};
        Image::new(xy{x: width as u32, y: height as u32}, Box::from(unsafe{std::slice::from_raw_parts(data as *const u16, (data_bytes/2) as usize)}))
    }
}
