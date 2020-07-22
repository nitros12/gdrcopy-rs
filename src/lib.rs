use cuda_sys::cuda::{
    cuPointerGetAttribute, CUdeviceptr, CUpointer_attribute, CUresult,
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS,
};
use libc::c_ulong;
use rustacuda::memory::DeviceBuffer;
use std::{
    ffi::c_void,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    slice,
};

pub mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct GDR {
    gdr: NonNull<ffi::gdr>,
}

impl GDR {
    pub fn open() -> Option<GDR> {
        let gdr = unsafe { ffi::gdr_open() };
        NonNull::new(gdr).map(|p| GDR { gdr: p })
    }

    pub fn map_device_buffer<'a, 'handle: 'a, 'buffer: 'a>(
        &'handle self,
        buffer: &'buffer mut DeviceBuffer<u8>,
    ) -> Option<GDRMap<'a>> {
        let device_addr = buffer.as_device_ptr().as_raw_mut() as usize;
        let len = buffer.len();

        let mut tokens = MaybeUninit::<CUDA_POINTER_ATTRIBUTE_P2P_TOKENS>::uninit();

        if CUresult::CUDA_SUCCESS
            != unsafe {
                cuPointerGetAttribute(
                    tokens.as_mut_ptr() as *mut c_void,
                    CUpointer_attribute::CU_POINTER_ATTRIBUTE_P2P_TOKENS,
                    device_addr as CUdeviceptr,
                )
            }
        {
            return None;
        }

        let tokens = unsafe { tokens.assume_init() };

        self.map(device_addr, len, tokens.p2pToken, tokens.vaSpaceToken)
    }

    pub fn map(
        &self,
        device_addr: usize,
        len: usize,
        p2p_token: u64,
        va_space: u32,
    ) -> Option<GDRMap> {
        let mut g = self.gdr;
        let mut handle = MaybeUninit::<ffi::gdr_mh_s>::uninit();

        // align to gpu page
        let len = (len + ffi::GPU_PAGE_SIZE as usize - 1) & ffi::GPU_PAGE_MASK as usize;

        if 0 != unsafe {
            ffi::gdr_pin_buffer(
                g.as_mut(),
                device_addr as c_ulong,
                len as u64,
                p2p_token,
                va_space,
                handle.as_mut_ptr(),
            )
        } {
            return None;
        }

        let handle = unsafe { handle.assume_init() };
        let mut p = MaybeUninit::<*mut c_void>::uninit();

        if 0 != unsafe { ffi::gdr_map(g.as_mut(), handle, p.as_mut_ptr(), len as u64) } {
            unsafe { ffi::gdr_unpin_buffer(g.as_mut(), handle) };
            return None;
        }

        let buf = unsafe { p.assume_init() as *mut u8 };

        Some(GDRMap {
            gdr: self,
            handle,
            buf,
            len,
        })
    }
}

impl Drop for GDR {
    fn drop(&mut self) {
        unsafe { ffi::gdr_close(self.gdr.as_mut()) };
    }
}

pub struct GDRMap<'handle> {
    gdr: &'handle GDR,
    handle: ffi::gdr_mh_s,
    buf: *mut u8,
    len: usize,
}

impl<'handle> Deref for GDRMap<'handle> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.buf as *const u8, self.len) }
    }
}

impl<'handle> DerefMut for GDRMap<'handle> {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.buf, self.len) }
    }
}

impl<'handle> AsRef<[u8]> for GDRMap<'handle> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl<'handle> AsMut<[u8]> for GDRMap<'handle> {
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl<'handle> Drop for GDRMap<'handle> {
    fn drop(&mut self) {
        let mut g = self.gdr.gdr;
        if 0 != unsafe {
            ffi::gdr_unmap(
                g.as_mut(),
                self.handle,
                self.buf as *mut c_void,
                self.len as u64,
            )
        } {
            return;
        }

        unsafe {
            ffi::gdr_unpin_buffer(g.as_mut(), self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ffi::*;

    #[test]
    fn test_open() {
        let gdr = unsafe { gdr_open() };
        assert!(!gdr.is_null());
        unsafe { gdr_close(gdr) };
    }
}
