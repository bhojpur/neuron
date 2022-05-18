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
	"encoding/binary"
	"hash/fnv"
)

// hashIntArray uses fnv to generate an int
func hashIntArray(in []int) int {
	tmp := make([]byte, 8*len(in))
	for i := 0; i < len(in); i++ {
		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in[i]))
	}
	h := fnv.New64a()
	v, _ := h.Write(tmp)
	return v
}

// func hashIntArrayPair(in1, in2 []int) int {
// 	n := len(in1) + len(in2)
// 	tmp := make([]byte, 8*n)
// 	i := 0
// 	for ; i < len(in1); i++ {
// 		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in1[i]))
// 	}
// 	for j := 0; j < len(in2); j++ {
// 		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in2[j]))
// 		i++
// 	}
// 	h := fnv.New64a()
// 	v, _ := h.Write(tmp)
// 	return v
// }
