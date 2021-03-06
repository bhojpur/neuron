// Code generated by Bhojpur Neuron (neurgen). DO NOT EDIT.

package tensor

import (
	"fmt"
	"math"
	"math/cmplx"
	"unsafe"

	"github.com/bhojpur/neuron/pkg/math32"
)

func anyToFloat64s(x interface{}) (retVal []float64) {
	switch xt := x.(type) {
	case []int:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int8:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int16:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint8:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint16:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []float32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case math32.IsNaN(v):
				retVal[i] = math.NaN()
			case math32.IsInf(v, 1):
				retVal[i] = math.Inf(1)
			case math32.IsInf(v, -1):
				retVal[i] = math.Inf(-1)
			default:
				retVal[i] = float64(v)
			}
		}
		return
	case []float64:
		return xt
	case []complex64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case cmplx.IsNaN(complex128(v)):
				retVal[i] = math.NaN()
			case cmplx.IsInf(complex128(v)):
				retVal[i] = math.Inf(1)
			default:
				retVal[i] = float64(real(v))
			}
		}
		return
	case []complex128:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case cmplx.IsNaN(v):
				retVal[i] = math.NaN()
			case cmplx.IsInf(v):
				retVal[i] = math.Inf(1)
			default:
				retVal[i] = real(v)
			}
		}
		return
	}
	panic("Unreachable")
}

func identityVal(x int, dt Dtype) interface{} {
	switch dt {
	case Int:
		return int(x)
	case Int8:
		return int8(x)
	case Int16:
		return int16(x)
	case Int32:
		return int32(x)
	case Int64:
		return int64(x)
	case Uint:
		return uint(x)
	case Uint8:
		return uint8(x)
	case Uint16:
		return uint16(x)
	case Uint32:
		return uint32(x)
	case Uint64:
		return uint64(x)
	case Float32:
		return float32(x)
	case Float64:
		return float64(x)
	case Complex64:
		var c complex64
		if x == 0 {
			return c
		}
		c = 1
		return c
	case Complex128:
		var c complex128
		if x == 0 {
			return c
		}
		c = 1
		return c
	case Bool:
		if x == 0 {
			return false
		}
		return true
	case String:
		if x == 0 {
			return ""
		}
		return fmt.Sprintf("%v", x)
	default:
		return x
	}
}
func threewayEq(a, b, c interface{}) bool {
	switch at := a.(type) {
	case []int:
		bt := b.([]int)
		ct := c.([]int)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []int8:
		bt := b.([]int8)
		ct := c.([]int8)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []int16:
		bt := b.([]int16)
		ct := c.([]int16)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []int32:
		bt := b.([]int32)
		ct := c.([]int32)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []int64:
		bt := b.([]int64)
		ct := c.([]int64)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []uint:
		bt := b.([]uint)
		ct := c.([]uint)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []uint8:
		bt := b.([]uint8)
		ct := c.([]uint8)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []uint16:
		bt := b.([]uint16)
		ct := c.([]uint16)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []uint32:
		bt := b.([]uint32)
		ct := c.([]uint32)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []uint64:
		bt := b.([]uint64)
		ct := c.([]uint64)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []float32:
		bt := b.([]float32)
		ct := c.([]float32)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []float64:
		bt := b.([]float64)
		ct := c.([]float64)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []complex64:
		bt := b.([]complex64)
		ct := c.([]complex64)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case []complex128:
		bt := b.([]complex128)
		ct := c.([]complex128)

		for i, va := range at {
			if va == 1 && bt[i] == 1 {
				if ct[i] != 1 {
					return false
				}
			}
		}
		return true
	case int:
		bt := b.(int)
		ct := c.(int)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case int8:
		bt := b.(int8)
		ct := c.(int8)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case int16:
		bt := b.(int16)
		ct := c.(int16)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case int32:
		bt := b.(int32)
		ct := c.(int32)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case int64:
		bt := b.(int64)
		ct := c.(int64)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case uint:
		bt := b.(uint)
		ct := c.(uint)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case uint8:
		bt := b.(uint8)
		ct := c.(uint8)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case uint16:
		bt := b.(uint16)
		ct := c.(uint16)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case uint32:
		bt := b.(uint32)
		ct := c.(uint32)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case uint64:
		bt := b.(uint64)
		ct := c.(uint64)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case float32:
		bt := b.(float32)
		ct := c.(float32)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case float64:
		bt := b.(float64)
		ct := c.(float64)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case complex64:
		bt := b.(complex64)
		ct := c.(complex64)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	case complex128:
		bt := b.(complex128)
		ct := c.(complex128)
		if (at == 1 && bt == 1) && ct != 1 {
			return false
		}
		return true
	}

	return false
}

func identityB(a bool) bool                                 { return a }
func identityI(a int) int                                   { return a }
func identityI8(a int8) int8                                { return a }
func identityI16(a int16) int16                             { return a }
func identityI32(a int32) int32                             { return a }
func identityI64(a int64) int64                             { return a }
func identityU(a uint) uint                                 { return a }
func identityU8(a uint8) uint8                              { return a }
func identityU16(a uint16) uint16                           { return a }
func identityU32(a uint32) uint32                           { return a }
func identityU64(a uint64) uint64                           { return a }
func identityUintptr(a uintptr) uintptr                     { return a }
func identityF32(a float32) float32                         { return a }
func identityF64(a float64) float64                         { return a }
func identityC64(a complex64) complex64                     { return a }
func identityC128(a complex128) complex128                  { return a }
func identityStr(a string) string                           { return a }
func identityUnsafePointer(a unsafe.Pointer) unsafe.Pointer { return a }
func mutateB(a bool) bool                                   { return true }
func mutateI(a int) int                                     { return 1 }
func mutateI8(a int8) int8                                  { return 1 }
func mutateI16(a int16) int16                               { return 1 }
func mutateI32(a int32) int32                               { return 1 }
func mutateI64(a int64) int64                               { return 1 }
func mutateU(a uint) uint                                   { return 1 }
func mutateU8(a uint8) uint8                                { return 1 }
func mutateU16(a uint16) uint16                             { return 1 }
func mutateU32(a uint32) uint32                             { return 1 }
func mutateU64(a uint64) uint64                             { return 1 }
func mutateUintptr(a uintptr) uintptr                       { return 0xdeadbeef }
func mutateF32(a float32) float32                           { return 1 }
func mutateF64(a float64) float64                           { return 1 }
func mutateC64(a complex64) complex64                       { return 1 }
func mutateC128(a complex128) complex128                    { return 1 }
func mutateStr(a string) string                             { return "Hello World" }
func mutateUnsafePointer(a unsafe.Pointer) unsafe.Pointer   { return unsafe.Pointer(uintptr(0xdeadbeef)) }
