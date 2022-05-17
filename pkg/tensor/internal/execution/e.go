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
	"fmt"
	"reflect"
	"unsafe"

	"github.com/bhojpur/neuron/pkg/tensor/internal/storage"
)

// E is the standard engine. It's to be embedded in package tensor
type E struct{}

// basic types supported.
var (
	Bool       = reflect.TypeOf(true)
	Int        = reflect.TypeOf(int(1))
	Int8       = reflect.TypeOf(int8(1))
	Int16      = reflect.TypeOf(int16(1))
	Int32      = reflect.TypeOf(int32(1))
	Int64      = reflect.TypeOf(int64(1))
	Uint       = reflect.TypeOf(uint(1))
	Uint8      = reflect.TypeOf(uint8(1))
	Uint16     = reflect.TypeOf(uint16(1))
	Uint32     = reflect.TypeOf(uint32(1))
	Uint64     = reflect.TypeOf(uint64(1))
	Float32    = reflect.TypeOf(float32(1))
	Float64    = reflect.TypeOf(float64(1))
	Complex64  = reflect.TypeOf(complex64(1))
	Complex128 = reflect.TypeOf(complex128(1))
	String     = reflect.TypeOf("")

	// aliases
	Byte = Uint8

	// extras
	Uintptr       = reflect.TypeOf(uintptr(0))
	UnsafePointer = reflect.TypeOf(unsafe.Pointer(&Uintptr))
)

func isScalar(a *storage.Header, t reflect.Type) bool { return a.TypedLen(t) == 1 }

type errorIndices []int

func (e errorIndices) Indices() []int { return []int(e) }
func (e errorIndices) Error() string  { return fmt.Sprintf("Error in indices %v", []int(e)) }

const (
	lenMismatch        = `Cannot compare with differing lengths: %d and %d`
	reductionErrMsg    = `Cannot reduce with function of type %T`
	defaultValueErrMsg = `Expected default value of type %T. Got %v of %T instead`
	typeMismatch       = `TypeMismatch: a %T and b %T`
)
