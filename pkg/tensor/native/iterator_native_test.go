// Code generated by Bhojpur Neuron. DO NOT EDIT.

package native

import (
	"testing"

	"github.com/stretchr/testify/assert"
	. "github.com/bhojpur/neuron/pkg/tensor"
)

func Test_VectorB(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(Bool), WithShape(6))
	it, err := VectorB(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixB(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(Bool), WithShape(2, 3))
	it, err := MatrixB(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3B(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(Bool), WithShape(2, 3, 4))
	it, err := Tensor3B(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorI(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int, 0, 6)), WithShape(6))
	it, err := VectorI(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixI(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int, 0, 6)), WithShape(2, 3))
	it, err := MatrixI(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3I(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3I(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorI8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int8, 0, 6)), WithShape(6))
	it, err := VectorI8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixI8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int8, 0, 6)), WithShape(2, 3))
	it, err := MatrixI8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3I8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int8, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3I8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorI16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int16, 0, 6)), WithShape(6))
	it, err := VectorI16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixI16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int16, 0, 6)), WithShape(2, 3))
	it, err := MatrixI16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3I16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int16, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3I16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorI32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int32, 0, 6)), WithShape(6))
	it, err := VectorI32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixI32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int32, 0, 6)), WithShape(2, 3))
	it, err := MatrixI32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3I32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int32, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3I32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorI64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int64, 0, 6)), WithShape(6))
	it, err := VectorI64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixI64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int64, 0, 6)), WithShape(2, 3))
	it, err := MatrixI64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3I64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Int64, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3I64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorU(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint, 0, 6)), WithShape(6))
	it, err := VectorU(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixU(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint, 0, 6)), WithShape(2, 3))
	it, err := MatrixU(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3U(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3U(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorU8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint8, 0, 6)), WithShape(6))
	it, err := VectorU8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixU8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint8, 0, 6)), WithShape(2, 3))
	it, err := MatrixU8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3U8(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint8, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3U8(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorU16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint16, 0, 6)), WithShape(6))
	it, err := VectorU16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixU16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint16, 0, 6)), WithShape(2, 3))
	it, err := MatrixU16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3U16(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint16, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3U16(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorU32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint32, 0, 6)), WithShape(6))
	it, err := VectorU32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixU32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint32, 0, 6)), WithShape(2, 3))
	it, err := MatrixU32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3U32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint32, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3U32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorU64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint64, 0, 6)), WithShape(6))
	it, err := VectorU64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixU64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint64, 0, 6)), WithShape(2, 3))
	it, err := MatrixU64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3U64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Uint64, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3U64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorF32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float32, 0, 6)), WithShape(6))
	it, err := VectorF32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixF32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float32, 0, 6)), WithShape(2, 3))
	it, err := MatrixF32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3F32(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3F32(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorF64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float64, 0, 6)), WithShape(6))
	it, err := VectorF64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixF64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float64, 0, 6)), WithShape(2, 3))
	it, err := MatrixF64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3F64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Float64, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3F64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorC64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex64, 0, 6)), WithShape(6))
	it, err := VectorC64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixC64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex64, 0, 6)), WithShape(2, 3))
	it, err := MatrixC64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3C64(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex64, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3C64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorC128(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex128, 0, 6)), WithShape(6))
	it, err := VectorC128(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixC128(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex128, 0, 6)), WithShape(2, 3))
	it, err := MatrixC128(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3C128(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(WithBacking(Range(Complex128, 0, 24)), WithShape(2, 3, 4))
	it, err := Tensor3C128(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}

func Test_VectorStr(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(String), WithShape(6))
	it, err := VectorStr(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
}

func Test_MatrixStr(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(String), WithShape(2, 3))
	it, err := MatrixStr(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
}

func Test_Tensor3Str(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	T = New(Of(String), WithShape(2, 3, 4))
	it, err := Tensor3Str(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
}