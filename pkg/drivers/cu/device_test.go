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

import (
	"bytes"
	"fmt"
	"testing"
)

func TestDevice(t *testing.T) {
	devices, err := NumDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices == 0 {
		return
	}

	buf := new(bytes.Buffer)

	for id := 0; id < devices; id++ {
		d, err := GetDevice(id)
		if err != nil {
			t.Fatal(err)
		}

		name, err := d.Name()
		if err != nil {
			t.Fatal(err)
		}

		cr, err := d.Attribute(ClockRate)
		if err != nil {
			t.Fatal(err)
		}

		mem, err := d.TotalMem()
		if err != nil {
			t.Fatal(err)
		}

		maj, err := d.Attribute(ComputeCapabilityMajor)
		if err != nil {
			t.Fatal(err)
		}

		min, err := d.Attribute(ComputeCapabilityMinor)
		if err != nil {
			t.Fatal(err)
		}
		uuid, err := d.UUID()
		if err != nil {
			t.Fatal(err)
		}

		fmt.Fprintf(buf, "Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Fprintf(buf, "Clock Rate:\t%v kHz\n", cr)
		fmt.Fprintf(buf, "Memory    :\t%v bytes\n", mem)
		fmt.Fprintf(buf, "Compute   :\t%d.%d\n", maj, min)
		fmt.Fprintf(buf, "UUID      :\t%v\n", uuid)
		t.Log(buf.String())

		buf.Reset()
	}
}

func TestVersion(t *testing.T) {
	t.Logf("CUDA Toolkit version: %v", Version())
}
