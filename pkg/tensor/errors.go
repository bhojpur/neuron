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

import "fmt"

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	NoOp() bool
}

// MathError is an error that occurs in an Array. It lists the indices for which an error has happened
type MathError interface {
	Indices() []int
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

func handleNoOp(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := err.(NoOpError); ok {
		return nil
	}
	return err
}

type errorIndices []int

func (e errorIndices) Indices() []int { return []int(e) }
func (e errorIndices) Error() string  { return fmt.Sprintf("Error in indices %v", []int(e)) }

const (
	emptyTensor       = "Tensor is uninitialized (no shape, no data)"
	dimMismatch       = "Dimension mismatch. Expected %d, got %d"
	atleastDims       = "Tensor has to be at least %d dimensions"
	dtypeMismatch     = "Dtype mismatch. Expected %v. Got %v"
	indexOOBAxis      = "Index %d is out of bounds for axis %d which has size %d"
	invalidAxis       = "Invalid axis %d for ndarray with %d dimensions"
	repeatedAxis      = "repeated axis %d in permutation pattern"
	invalidSliceIndex = "Invalid slice index. Start: %d, End: %d"
	sliceIndexOOB     = "Slice index out of bounds: Start: %d, End: %d. Length: %d"
	broadcastError    = "Cannot broadcast together. Resulting shape will be at least (%d, 1). Repeats is (%d, 1)"
	lenMismatch       = "Cannot compare with differing lengths: %d and %d"
	typeMismatch      = "TypeMismatch: a %v and b %v"
	typeclassMismatch = "Typeclass mismatch on %v"
	shapeMismatch     = "Shape mismatch. Expected %v. Got %v"
	sizeMismatch      = "Size Mismatch. %d and %d"
	reuseReshapeErr   = "Failed to reshape the reuse *Dense from into %v. Size was: %d"
	incrReshapeErr    = "Failed to reshape the incr *Dense into %v. Size was: %d"
	retValReshapeErr  = "Failed to reshape the retVal *Dense into %v. Size was: %d"
	div0              = "Division by 0. Index was %v"
	div0General       = "Division by 0"
	opFail            = "Failed to perform %v"
	extractionFail    = "Failed to extract %v from %T"
	unknownState      = "Unknown state reached: Safe %t, Incr %t, Reuse %t"
	unsupportedDtype  = "Array of %v is unsupported for %v"
	maskRequired      = "Masked array type required for %v"
	inaccessibleData  = "Data in %p inaccessible"

	methodNYI = "%q not yet implemented for %v"
	typeNYI   = "%q not yet implemented for interactions with %T"
)
