package tensor

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
	"encoding/gob"
	"fmt"
	"io"

	"github.com/pkg/errors"
)

func (t *CS) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	if err = encoder.Encode(t.s); err != nil {
		return
	}

	if err = encoder.Encode(t.o); err != nil {
		return
	}

	if err = encoder.Encode(t.indices); err != nil {
		return
	}

	if err = encoder.Encode(t.indptr); err != nil {
		return
	}

	data := t.Data()
	if err = encoder.Encode(&data); err != nil {
		return
	}

	return buf.Bytes(), nil
}

func (t *CS) GobDecode(p []byte) (err error) {
	buf := bytes.NewBuffer(p)
	decoder := gob.NewDecoder(buf)

	var shape Shape
	if err = decoder.Decode(&shape); err != nil {
		return
	}
	t.s = shape

	var o DataOrder
	if err = decoder.Decode(&o); err != nil {
		return
	}

	var indices []int
	if err = decoder.Decode(&indices); err != nil {
		return
	}
	t.indices = indices

	var indptr []int
	if err = decoder.Decode(&indptr); err != nil {
		return
	}
	t.indptr = indptr

	var data interface{}
	if err = decoder.Decode(&data); err != nil {
		return
	}
	t.array = arrayFromSlice(data)
	return nil
}

func (t *CS) WriteNpy(w io.Writer) error { return errors.Errorf("Cannot write to Npy") }
func (t *CS) ReadNpy(r io.Reader) error  { return errors.Errorf("Cannot read from npy") }
func (t *CS) Format(s fmt.State, c rune) {}
func (t *CS) String() string             { return "CS" }
