package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// This file contains the tests for API functions that aren't generated by Bhojpur Neuron

func TestLtScalarScalar(t *testing.T) {
	// scalar-scalar
	a := New(WithBacking([]float64{6}))
	b := New(WithBacking([]float64{2}))
	var correct interface{} = false

	res, err := Lt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// scalar-tensor
	a = New(WithBacking([]float64{1, 4}))
	b = New(WithBacking([]float64{2}))
	correct = []bool{true, false}

	res, err = Lt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor-scalar
	a = New(WithBacking([]float64{3}))
	b = New(WithBacking([]float64{6, 2}))
	correct = []bool{true, false}

	res, err = Lt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor - tensor
	a = New(WithBacking([]float64{21, 2}))
	b = New(WithBacking([]float64{7, 10}))
	correct = []bool{false, true}

	res, err = Lt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
}

func TestGtScalarScalar(t *testing.T) {
	// scalar-scalar
	a := New(WithBacking([]float64{6}))
	b := New(WithBacking([]float64{2}))
	var correct interface{} = true

	res, err := Gt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// scalar-tensor
	a = New(WithBacking([]float64{1, 4}))
	b = New(WithBacking([]float64{2}))
	correct = []bool{false, true}

	res, err = Gt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor-scalar
	a = New(WithBacking([]float64{3}))
	b = New(WithBacking([]float64{6, 2}))
	correct = []bool{false, true}

	res, err = Gt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor - tensor
	a = New(WithBacking([]float64{21, 2}))
	b = New(WithBacking([]float64{7, 10}))
	correct = []bool{true, false}

	res, err = Gt(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
}

func TestLteScalarScalar(t *testing.T) {
	// scalar-scalar
	a := New(WithBacking([]float64{6}))
	b := New(WithBacking([]float64{2}))
	var correct interface{} = false

	res, err := Lte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// scalar-tensor
	a = New(WithBacking([]float64{1, 2, 4}))
	b = New(WithBacking([]float64{2}))
	correct = []bool{true, true, false}

	res, err = Lte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor-scalar
	a = New(WithBacking([]float64{3}))
	b = New(WithBacking([]float64{6, 2}))
	correct = []bool{true, false}

	res, err = Lte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor - tensor
	a = New(WithBacking([]float64{21, 2}))
	b = New(WithBacking([]float64{7, 10}))
	correct = []bool{false, true}

	res, err = Lte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
}

func TestGteScalarScalar(t *testing.T) {
	// scalar-scalar
	a := New(WithBacking([]float64{6}))
	b := New(WithBacking([]float64{2}))
	var correct interface{} = true

	res, err := Gte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// scalar-tensor
	a = New(WithBacking([]float64{1, 2, 4}))
	b = New(WithBacking([]float64{2}))
	correct = []bool{false, true, true}

	res, err = Gte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor-scalar
	a = New(WithBacking([]float64{3}))
	b = New(WithBacking([]float64{6, 3, 2}))
	correct = []bool{false, true, true}

	res, err = Gte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor - tensor
	a = New(WithBacking([]float64{21, 31, 2}))
	b = New(WithBacking([]float64{7, 31, 10}))
	correct = []bool{true, true, false}

	res, err = Gte(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
}

func TestElEqScalarScalar(t *testing.T) {
	// scalar-scalar
	a := New(WithBacking([]float64{6}))
	b := New(WithBacking([]float64{2}))
	var correct interface{} = false

	res, err := ElEq(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// scalar-tensor
	a = New(WithBacking([]float64{1, 2, 4}))
	b = New(WithBacking([]float64{2}))
	correct = []bool{false, true, false}

	res, err = ElEq(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor-scalar
	a = New(WithBacking([]float64{3}))
	b = New(WithBacking([]float64{6, 3, 2}))
	correct = []bool{false, true, false}

	res, err = ElEq(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())

	// tensor - tensor
	a = New(WithBacking([]float64{21, 10}))
	b = New(WithBacking([]float64{7, 10}))
	correct = []bool{false, true}

	res, err = ElEq(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
}