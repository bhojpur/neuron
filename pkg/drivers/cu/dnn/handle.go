package cudnn

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

// #include <cudnn.h>
import "C"

// Memory represents an instance of CUDA memory
type Memory interface {
	Uintptr() uintptr

	IsNativelyAccessible() bool
}

// Context represents the context in which cuDNN operations are performed in.
//
// Internally, the Context holds a cudnnHandle_t
//
// Once the context has been finished, do remember to call `Close` on the context.
type Context struct {
	internal C.cudnnHandle_t
}

// NewContext creates a new Context. This is the only function that will panic if it is unable to create the context.
func NewContext() (retVal *Context) {
	var internal C.cudnnHandle_t
	if err := result(C.cudnnCreate(&internal)); err != nil {
		panic(err)
	}
	retVal = &Context{internal}
	return retVal
}

//  Close destroys the underlying context.
func (ctx *Context) Close() error {
	var empty C.cudnnHandle_t
	if ctx.internal == empty {
		return nil
	}

	if err := result(C.cudnnDestroy(ctx.internal)); err != nil {
		return err
	}
	ctx.internal = empty
	return nil
}
