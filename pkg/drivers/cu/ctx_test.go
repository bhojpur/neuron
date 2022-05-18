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

import "testing"

func TestContext(t *testing.T) {
	ctx := NewContext(Device(0), SchedAuto)
	mem, err := ctx.MemAlloc(24)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", mem)
}

func TestMultipleContext(t *testing.T) {
	d := Device(0)
	ctx0 := NewManuallyManagedContext(d, SchedAuto)
	ctx1 := NewManuallyManagedContext(d, SchedAuto)

	errChan0 := make(chan error)
	errChan1 := make(chan error)
	go ctx0.Run(errChan0)
	go ctx1.Run(errChan1)

	if err := <-errChan0; err != nil {
		t.Fatalf("err while initializing run of ctx0 %v", err)
	}
	if err := <-errChan1; err != nil {
		t.Fatalf("err while initializing run of ctx1 %v", err)
	}

	var mem0, mem1 DevicePtr
	var err error
	if mem0, err = ctx0.MemAlloc(1024); err != nil {
		t.Errorf("Err while alloc in ctx0: %v", err)
	}

	if mem1, err = ctx1.MemAlloc(1024); err != nil {
		t.Errorf("Err while alloc in ctx1: %v", err)
	}

	t.Logf("Mem0: %v", mem0)
	t.Logf("Mem1: %v", mem1)
	ctx0.MemFree(mem0)
	ctx1.MemFree(mem1)

	if err = ctx0.Error(); err != nil {
		t.Errorf("Error while freeing %v", err)
	}
	if err = ctx1.Error(); err != nil {
		t.Errorf("Error while freeing %v", err)
	}

	// runtime.GC()
}
