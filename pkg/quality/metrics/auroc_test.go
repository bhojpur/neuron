package metrics

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
	"math"
	"testing"
)

func TestAreaUnderROCComputation(t *testing.T) {

	//the area under the ROC curve function should compute the area confirmed by known good sources.
	target := []float64{1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
		0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
		1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
		0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0}
	predictor := []float64{0.5235577, 0.30661483, 0.59139475, 0.37892545, 0.05816997,
		0.94734317, 0.66138486, 0.54269425, 0.21452441, 0.94896081,
		0.41993908, 0.58837957, 0.11408175, 0.48312741, 0.84003565,
		0.33945467, 0.11559768, 0.71648437, 0.21810821, 0.92488452,
		0.05453535, 0.44637233, 0.64535999, 0.77394022, 0.9026129,
		0.28982892, 0.6794064, 0.61858061, 0.45078173, 0.90522972,
		0.04655416, 0.77567769, 0.8518629, 0.74771238, 0.85526817,
		0.5229841, 0.96954495, 0.6240129, 0.82430844, 0.18066989,
		0.22395191, 0.21699929, 0.8894576, 0.43751175, 0.89785098,
		0.79745987, 0.87576878, 0.84688697, 0.05700901, 0.05030223,
		0.36726464, 0.3322127, 0.2732018, 0.14278569, 0.32005824,
		0.81209509, 0.44409538, 0.08827284, 0.70502815, 0.55366671,
		0.16181728, 0.56807671, 0.8449648, 0.59481691, 0.34040533,
		0.80539343, 0.62812843, 0.51990045, 0.73393991, 0.58840707,
		0.03312294, 0.30291744, 0.07525649, 0.61507223, 0.90573683,
		0.20743752, 0.61830969, 0.03828613, 0.91732371, 0.07572959,
		0.15598528, 0.70567578, 0.12247885, 0.77424471, 0.19348568,
		0.21060475, 0.60223579, 0.21299479, 0.12266292, 0.66244018,
		0.80382931, 0.31061611, 0.47330931, 0.1956672, 0.86662296,
		0.78720881, 0.20607393, 0.78361461, 0.5680686, 0.26827603}

	//auc1, auc0, u1, u0, err := AreaUnderROC(predictor, target, nil)
	slcroc, err := AreaUnderROC(predictor, target, nil)
	if err != nil {
		t.Fatal(err)
	}
	roc := slcroc[0]
	auc1 := roc.Auc1
	auc0 := roc.Auc0
	u1 := roc.U1
	u0 := roc.U0

	if math.Abs(auc1-0.8348) > 1e-6 {
		t.Fatal("bad auc1")
	}
	if math.Abs(auc0-0.1652) > 1e-6 {
		t.Fatal("bad auc0")
	}
	if math.Abs(u1-2087.0) > 1e-6 {
		t.Fatal("bad u1")
	}
	if math.Abs(u0-413.0) > 1e-6 {
		t.Fatal("bad u0")
	}
	//  // verified auc1 in R with
	// > library(pROC)
	// > category=c(1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
	// 			0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
	// 			1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
	// 			1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
	// 			0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0)
	// > prediction=c(0.5235577, 0.30661483, 0.59139475, 0.37892545, 0.05816997,
	// 			0.94734317, 0.66138486, 0.54269425, 0.21452441, 0.94896081,
	// 			0.41993908, 0.58837957, 0.11408175, 0.48312741, 0.84003565,
	// 			0.33945467, 0.11559768, 0.71648437, 0.21810821, 0.92488452,
	// 			0.05453535, 0.44637233, 0.64535999, 0.77394022, 0.9026129,
	// 			0.28982892, 0.6794064, 0.61858061, 0.45078173, 0.90522972,
	// 			0.04655416, 0.77567769, 0.8518629, 0.74771238, 0.85526817,
	// 			0.5229841, 0.96954495, 0.6240129, 0.82430844, 0.18066989,
	// 			0.22395191, 0.21699929, 0.8894576, 0.43751175, 0.89785098,
	// 			0.79745987, 0.87576878, 0.84688697, 0.05700901, 0.05030223,
	// 			0.36726464, 0.3322127, 0.2732018, 0.14278569, 0.32005824,
	// 			0.81209509, 0.44409538, 0.08827284, 0.70502815, 0.55366671,
	// 			0.16181728, 0.56807671, 0.8449648, 0.59481691, 0.34040533,
	// 			0.80539343, 0.62812843, 0.51990045, 0.73393991, 0.58840707,
	// 			0.03312294, 0.30291744, 0.07525649, 0.61507223, 0.90573683,
	// 			0.20743752, 0.61830969, 0.03828613, 0.91732371, 0.07572959,
	// 			0.15598528, 0.70567578, 0.12247885, 0.77424471, 0.19348568,
	// 			0.21060475, 0.60223579, 0.21299479, 0.12266292, 0.66244018,
	// 			0.80382931, 0.31061611, 0.47330931, 0.1956672, 0.86662296,
	// 			0.78720881, 0.20607393, 0.78361461, 0.5680686, 0.26827603)
	// > roc_obj <- roc(category, prediction)
	// Setting levels: control = 0, case = 1
	// Setting direction: controls < cases
	// > auc(roc_obj)
	// Area under the curve: 0.8348
	// >
}

func TestAreaUnderROCComputation2(t *testing.T) {

	// the area under the ROC curve function should compute the area confirmed by known good sources. very small example.

	target := []float64{1, 0, 1, 0, 0}
	predictor := []float64{0.5235577, 0.30661483, 0.59139475, 0.37892545, 0.05816997}

	slcroc, err := AreaUnderROC(predictor, target, nil)
	if err != nil {
		t.Fatal(err)
	}
	roc := slcroc[0]
	auc1 := roc.Auc1
	auc0 := roc.Auc0
	u1 := roc.U1
	u0 := roc.U0

	if auc1 != 1.0 {
		t.Fatal("bad auc1")
	}
	if auc0 != 0.0 {
		t.Fatal("bad auc0")
	}
	if u1 != 6.0 {
		t.Fatal("bad u1")
	}
	if u0 != 0.0 {
		t.Fatal("bad u0")
	}
}
