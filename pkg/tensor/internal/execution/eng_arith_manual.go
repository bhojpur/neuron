package execution

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import (
	"reflect"

	"github.com/bhojpur/neuron/pkg/tensor/internal/storage"
	"github.com/pkg/errors"
)

func (e E) AddSliced(t reflect.Type, dataA *storage.Header, dstStart, dstEnd int, dataB *storage.Header, srcStart, srcEnd int) (err error) {
	ds := dstStart * int(t.Size())
	de := dstEnd * int(t.Size())
	a := &storage.Header{
		Raw: dataA.Raw[ds:de],
	}

	ss := srcStart * int(t.Size())
	se := srcEnd * int(t.Size())
	b := &storage.Header{
		Raw: dataB.Raw[ss:se],
	}

	as := isScalar(a, t)
	bs := isScalar(b, t)

	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()

		switch {
		case as && bs:
			VecAddI(at, bt)
		case as && !bs:
			AddSVI(at[0], bt)
		case !as && bs:
			AddVSI(at, bt[0])
		default:
			VecAddI(at, bt)
		}
		return
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()
		switch {
		case as && bs:
			VecAddI8(at, bt)
		case as && !bs:
			AddSVI8(at[0], bt)
		case !as && bs:
			AddVSI8(at, bt[0])
		default:
			VecAddI8(at, bt)
		}
		return
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()
		switch {
		case as && bs:
			VecAddI16(at, bt)
		case as && !bs:
			AddSVI16(at[0], bt)
		case !as && bs:
			AddVSI16(at, bt[0])
		default:
			VecAddI16(at, bt)
		}
		return
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()
		switch {
		case as && bs:
			VecAddI32(at, bt)
		case as && !bs:
			AddSVI32(at[0], bt)
		case !as && bs:
			AddVSI32(at, bt[0])
		default:
			VecAddI32(at, bt)
		}
		return
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()
		switch {
		case as && bs:
			VecAddI64(at, bt)
		case as && !bs:
			AddSVI64(at[0], bt)
		case !as && bs:
			AddVSI64(at, bt[0])
		default:
			VecAddI64(at, bt)
		}
		return
	case Uint:
		at := a.Uints()
		bt := b.Uints()
		switch {
		case as && bs:
			VecAddU(at, bt)
		case as && !bs:
			AddSVU(at[0], bt)
		case !as && bs:
			AddVSU(at, bt[0])
		default:
			VecAddU(at, bt)
		}
		return
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()
		switch {
		case as && bs:
			VecAddU8(at, bt)
		case as && !bs:
			AddSVU8(at[0], bt)
		case !as && bs:
			AddVSU8(at, bt[0])
		default:
			VecAddU8(at, bt)
		}
		return
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()
		switch {
		case as && bs:
			VecAddU16(at, bt)
		case as && !bs:
			AddSVU16(at[0], bt)
		case !as && bs:
			AddVSU16(at, bt[0])
		default:
			VecAddU16(at, bt)
		}
		return
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()
		switch {
		case as && bs:
			VecAddU32(at, bt)
		case as && !bs:
			AddSVU32(at[0], bt)
		case !as && bs:
			AddVSU32(at, bt[0])
		default:
			VecAddU32(at, bt)
		}
		return
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()
		switch {
		case as && bs:
			VecAddU64(at, bt)
		case as && !bs:
			AddSVU64(at[0], bt)
		case !as && bs:
			AddVSU64(at, bt[0])
		default:
			VecAddU64(at, bt)
		}
		return
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()
		switch {
		case as && bs:
			VecAddF32(at, bt)
		case as && !bs:
			AddSVF32(at[0], bt)
		case !as && bs:
			AddVSF32(at, bt[0])
		default:
			VecAddF32(at, bt)
		}
		return
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()
		switch {
		case as && bs:
			VecAddF64(at, bt)
		case as && !bs:
			AddSVF64(at[0], bt)
		case !as && bs:
			AddVSF64(at, bt[0])
		default:
			VecAddF64(at, bt)
		}
		return
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()
		switch {
		case as && bs:
			VecAddC64(at, bt)
		case as && !bs:
			AddSVC64(at[0], bt)
		case !as && bs:
			AddVSC64(at, bt[0])
		default:
			VecAddC64(at, bt)
		}
		return
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()
		switch {
		case as && bs:
			VecAddC128(at, bt)
		case as && !bs:
			AddSVC128(at[0], bt)
		case !as && bs:
			AddVSC128(at, bt[0])
		default:
			VecAddC128(at, bt)
		}
		return
	case String:
		at := a.Strings()
		bt := b.Strings()
		switch {
		case as && bs:
			VecAddStr(at, bt)
		case as && !bs:
			AddSVStr(at[0], bt)
		case !as && bs:
			AddVSStr(at, bt[0])
		default:
			VecAddStr(at, bt)
		}
		return
	default:
		return errors.Errorf("Unsupported type %v for Add", t)
	}
}
