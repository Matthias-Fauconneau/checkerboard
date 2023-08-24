use uvc::*;
pub struct IR(*mut uvc_stream_handle_t);
use image::{Image, xy};
impl super::Camera for IR {
    fn new(_: &str, _: bool) -> Self {
        use std::ptr::null_mut;
        let mut uvc = null_mut();
        assert!(unsafe{uvc_init(&mut uvc as *mut _, null_mut())} >= 0);
        /*let mut devices : *mut *mut uvc_device_t = null_mut();
        assert!(unsafe{uvc_find_devices(uvc, &mut devices as *mut _, 0/*x0bda*/, 0/*x5840*/, std::ptr::null())} >= 0);
        for device in std::iter::successors(Some(devices), |devices| Some(unsafe{devices.add(1)})) {
            let device = unsafe{*device};
            if device.is_null() { break; }
            let mut device_descriptor : *mut uvc_device_descriptor_t = null_mut();
            assert!(unsafe{uvc_get_device_descriptor(device, &mut device_descriptor as &mut _)} >= 0);
            assert!(!device_descriptor.is_null());
            let device_descriptor = unsafe{*device_descriptor};
            if device_descriptor.idProduct != 0x5840 { continue; }
            /*println!("{:x} {:x} {:x}", device_descriptor.idVendor, device_descriptor.idProduct, device_descriptor.bcdUVC);
            if !device_descriptor.serialNumber.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.serialNumber)}); }
            if !device_descriptor.manufacturer.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.manufacturer)}); }
            if !device_descriptor.product.is_null() { println!("{:?}", unsafe{std::ffi::CStr::from_ptr(device_descriptor.product)}); }*/
            let mut device_handle = null_mut();
            assert!(unsafe{uvc_open(device, &mut device_handle as *mut _)} >= 0);*/
            //if (libusb_set_option(NULL, LIBUSB_OPTION_NO_DEVICE_DISCOVERY, NULL)) {		
	        //if (libusb_init(&usb_ctx)) {
		    //if (pthread_create(&usb_thread, NULL, usb_handle_events, (void *) this)) {
	        //if (uvc_init(&uvc_ctx, usb_ctx)) {
            let mut device_handle = null_mut();
            let fd = rustix::fs::open("/dev/bus/usb/003/019",  rustix::fs::OFlags::RWMODE, rustix::fs::Mode::empty()).unwrap(); // /!\ don't drop
            extern "C" {
                pub fn uvc_init(ctx: *mut *mut uvc_context_t, usb_ctx: *mut libusb_context) -> uvc_error_t;
                pub fn uvc_wrap(sys_dev:  std::os::fd::RawFd, context: *const uvc_context_t, device_handle: *mut *mut uvc_device_handle_t) -> uvc_error_t;
            }
            use rustix::fd::IntoRawFd;
            assert_eq!(unsafe{uvc_wrap(fd.into_raw_fd(), uvc, &mut device_handle as *mut _)}, 0);
	
            let format /*uvc_format_desc_t*/ = unsafe{*uvc_get_format_descs(device_handle)};
	    	assert_eq!(format.bDescriptorSubtype, uvc_vs_desc_subtype_UVC_VS_FORMAT_UNCOMPRESSED);
	        let frame = unsafe{*format.frame_descs};
	        //dbg!(frame.wWidth, frame.wHeight);
            let mut control = unsafe{std::mem::zeroed()};
            assert_eq!(unsafe{uvc_get_stream_ctrl_format_size(device_handle, &mut control as *mut _, uvc_frame_format_UVC_FRAME_FORMAT_ANY, frame.wWidth as i32, frame.wHeight as i32, 0/*25*/)}, 0);
            //let mut stream = null_mut();
            /*assert!(unsafe{uvc_stream_open_ctrl(device_handle, &mut stream as *mut _, &mut control as *mut _)} >= 0);
            assert!(unsafe{uvc_stream_start(stream, None, null_mut(), 0)} >= 0);*/
            extern "C" fn uvc_callback(_frame: *mut uvc_frame_t, _user_ptr: *mut std::ffi::c_void) { println!("uvc_callbaack"); }
            assert_eq!(unsafe{uvc_start_streaming(device_handle, &mut control as *mut _, Some(uvc_callback), std::ptr::null_mut(), 0)}, 0);
            {let e = unsafe{uvc_set_zoom_abs(device_handle, 0x8004/*Mode::Temperature*/)}; assert!(e >= 0, "{e:?}");}
            /*return Self(stream);
        }
       panic!("UVC IR camera 0x5840 not found");*/
       Self(std::ptr::null_mut())
    }
    fn next(&mut self) -> Image<Box<[u16]>> {
        use uvc::*;
        let mut frame : *mut uvc_frame_t = std::ptr::null_mut();
        assert!(unsafe{uvc_stream_get_frame(self.0, &mut frame as *mut _, 1000000)} >= 0);
        assert!(!frame.is_null());
        let uvc_frame_t{width, height, data, data_bytes, ..} = unsafe{*frame};
        let data = Box::<[u16]>::from(unsafe{std::slice::from_raw_parts(data as *const u16, (data_bytes/2) as usize)});
        println!("{:x?}", &data[((height-3)*width+40) as usize..][..16]);
        Image::new(xy{x: width as u32, y: height as u32}, data)
    }
}
