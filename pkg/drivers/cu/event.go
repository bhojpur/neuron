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

// Event represents a CUDA event
type Event struct {
	ev C.CUevent
}

func makeEvent(event C.CUevent) Event { return Event{event} }

func (e Event) c() C.CUevent { return e.ev }

func MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	err = result(C.cuEventCreate(&event.ev, CFlags))
	return
}

func DestroyEvent(event *Event) (err error) {
	err = result(C.cuEventDestroy(event.ev))
	*event = Event{}
	return
}

func (ctx *Ctx) MakeEvent(flags EventFlags) (event Event, err error) {
	CFlags := C.uint(flags)
	f := func() error { return result(C.cuEventCreate(&event.ev, CFlags)) }
	if err = ctx.Do(f); err != nil {
		err = errors.Wrap(err, "MakeEvent")
		return
	}
	return
}

func (ctx *Ctx) DestroyEvent(event *Event) {
	f := func() error { return result(C.cuEventDestroy(event.ev)) }
	ctx.err = ctx.Do(f)
	*event = Event{}
	return
}
