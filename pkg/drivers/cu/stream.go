package cu

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

// #include <cuda.h>
import "C"
import (
	"github.com/pkg/errors"
)

// Stream represents a CUDA stream.
type Stream struct {
	s C.CUstream
}

var NoStream = Stream{}

func makeStream(s C.CUstream) Stream { return Stream{s} }
func (s Stream) c() C.CUstream       { return s.s }

// C is the exported version of the c method
func (s Stream) C() C.CUstream { return s.c() }

// MakeStream creates a stream. The flags determines the behaviors of the stream.
func MakeStream(flags StreamFlags) (Stream, error) {
	var s Stream
	err := result(C.cuStreamCreate(&s.s, C.uint(flags)))
	return s, err
}

// MakeStreamWithPriority creates a stream with the given priority. The flags determines the behaviors of the stream.
// This API alters the scheduler priority of work in the stream. Work in a higher priority stream may preempt work already executing in a low priority stream.
//
// `priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority.
//
// The range of meaningful numerical priorities can be queried using `StreamPriorityRange`.
// If the specified priority is outside the numerical range returned by `StreamPriorityRange`,
// it will automatically be clamped to the lowest or the highest number in the range.
func MakeStreamWithPriority(priority int, flags StreamFlags) (Stream, error) {
	var s Stream
	err := result(C.cuStreamCreateWithPriority(&s.s, C.uint(flags), C.int(priority)))
	return s, err
}

// DestroyStream destroys the stream specified by hStream.
//
// In case the device is still doing work in the stream hStream when DestroyStrea() is called,
// the function will return immediately and the resources associated with hStream will be released automatically once the device has completed all work in hStream.
func (hStream *Stream) Destroy() error {
	err := result(C.cuStreamDestroy(hStream.s))
	*hStream = Stream{}
	return err
}

func (ctx *Ctx) MakeStream(flags StreamFlags) (stream Stream, err error) {
	var s Stream

	f := func() error { return result(C.cuStreamCreate(&s.s, C.uint(flags))) }
	if err = ctx.Do(f); err != nil {
		return s, errors.Wrap(err, "MakeStream")
	}
	return s, nil
}

func (ctx *Ctx) MakeStreamWithPriority(priority int, flags StreamFlags) (Stream, error) {
	var s Stream
	f := func() error { return result(C.cuStreamCreateWithPriority(&s.s, C.uint(flags), C.int(priority))) }
	if err := ctx.Do(f); err != nil {
		return s, errors.Wrap(err, "MakeStream With Priority")
	}
	return s, nil
}

func (ctx *Ctx) DestroyStream(hStream *Stream) {
	f := func() error { return result(C.cuStreamDestroy(hStream.s)) }
	ctx.err = ctx.Do(f)
}
