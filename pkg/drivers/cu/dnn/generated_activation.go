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

/* Generated by gencudnn. DO NOT EDIT */

// #include <cudnn.h>
import "C"
import "runtime"

// Activation is a representation of cudnnActivationDescriptor_t.
type Activation struct {
	internal C.cudnnActivationDescriptor_t

	mode       ActivationMode
	reluNanOpt NanPropagation
	coef       float64
}

// NewActivation creates a new Activation.
func NewActivation(mode ActivationMode, reluNanOpt NanPropagation, coef float64) (retVal *Activation, err error) {
	var internal C.cudnnActivationDescriptor_t
	if err := result(C.cudnnCreateActivationDescriptor(&internal)); err != nil {
		return nil, err
	}

	if err := result(C.cudnnSetActivationDescriptor(internal, mode.C(), reluNanOpt.C(), C.double(coef))); err != nil {
		return nil, err
	}

	retVal = &Activation{
		internal:   internal,
		mode:       mode,
		reluNanOpt: reluNanOpt,
		coef:       coef,
	}
	runtime.SetFinalizer(retVal, destroyActivation)
	return retVal, nil
}

// C returns the cgo representation.
func (a *Activation) C() C.cudnnActivationDescriptor_t { return a.internal }

// Mode returns the internal mode.
func (a *Activation) Mode() ActivationMode { return a.mode }

// ReluNanOpt returns the internal reluNanOpt.
func (a *Activation) ReluNanOpt() NanPropagation { return a.reluNanOpt }

// Coef returns the internal coef.
func (a *Activation) Coef() float64 { return a.coef }

func destroyActivation(obj *Activation) { C.cudnnDestroyActivationDescriptor(obj.internal) }