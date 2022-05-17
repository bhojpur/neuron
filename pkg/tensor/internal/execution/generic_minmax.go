// Code generated by Bhojpur Neuron (neurgen). DO NOT EDIT.

package execution

func VecMinI(a, b []int) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVI(a int, b []int) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSI(a []int, b int) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxI(a, b []int) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVI(a int, b []int) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSI(a []int, b int) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinI8(a, b []int8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVI8(a int8, b []int8) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSI8(a []int8, b int8) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxI8(a, b []int8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVI8(a int8, b []int8) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSI8(a []int8, b int8) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinI16(a, b []int16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVI16(a int16, b []int16) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSI16(a []int16, b int16) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxI16(a, b []int16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVI16(a int16, b []int16) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSI16(a []int16, b int16) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinI32(a, b []int32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVI32(a int32, b []int32) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSI32(a []int32, b int32) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxI32(a, b []int32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVI32(a int32, b []int32) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSI32(a []int32, b int32) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinI64(a, b []int64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVI64(a int64, b []int64) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSI64(a []int64, b int64) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxI64(a, b []int64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVI64(a int64, b []int64) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSI64(a []int64, b int64) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinU(a, b []uint) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVU(a uint, b []uint) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSU(a []uint, b uint) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxU(a, b []uint) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVU(a uint, b []uint) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSU(a []uint, b uint) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinU8(a, b []uint8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVU8(a uint8, b []uint8) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSU8(a []uint8, b uint8) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxU8(a, b []uint8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVU8(a uint8, b []uint8) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSU8(a []uint8, b uint8) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinU16(a, b []uint16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVU16(a uint16, b []uint16) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSU16(a []uint16, b uint16) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxU16(a, b []uint16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVU16(a uint16, b []uint16) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSU16(a []uint16, b uint16) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinU32(a, b []uint32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVU32(a uint32, b []uint32) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSU32(a []uint32, b uint32) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxU32(a, b []uint32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVU32(a uint32, b []uint32) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSU32(a []uint32, b uint32) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinU64(a, b []uint64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVU64(a uint64, b []uint64) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSU64(a []uint64, b uint64) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxU64(a, b []uint64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVU64(a uint64, b []uint64) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSU64(a []uint64, b uint64) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinF32(a, b []float32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVF32(a float32, b []float32) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSF32(a []float32, b float32) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxF32(a, b []float32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVF32(a float32, b []float32) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSF32(a []float32, b float32) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinF64(a, b []float64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVF64(a float64, b []float64) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSF64(a []float64, b float64) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxF64(a, b []float64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVF64(a float64, b []float64) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSF64(a []float64, b float64) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func VecMinStr(a, b []string) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
}

func MinSVStr(a string, b []string) {
	for i := range b {
		if a < b[i] {
			b[i] = a
		}
	}
}

func MinVSStr(a []string, b string) {
	for i := range a {
		if b < a[i] {
			a[i] = b
		}
	}
}

func VecMaxStr(a, b []string) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

func MaxSVStr(a string, b []string) {
	for i := range b {
		if a > b[i] {
			b[i] = a
		}
	}
}

func MaxVSStr(a []string, b string) {
	for i := range a {
		if b > a[i] {
			a[i] = b
		}
	}
}
func MinI(a, b int) (c int) {
	if a < b {
		return a
	}
	return b
}

func MaxI(a, b int) (c int) {
	if a > b {
		return a
	}
	return b
}
func MinI8(a, b int8) (c int8) {
	if a < b {
		return a
	}
	return b
}

func MaxI8(a, b int8) (c int8) {
	if a > b {
		return a
	}
	return b
}
func MinI16(a, b int16) (c int16) {
	if a < b {
		return a
	}
	return b
}

func MaxI16(a, b int16) (c int16) {
	if a > b {
		return a
	}
	return b
}
func MinI32(a, b int32) (c int32) {
	if a < b {
		return a
	}
	return b
}

func MaxI32(a, b int32) (c int32) {
	if a > b {
		return a
	}
	return b
}
func MinI64(a, b int64) (c int64) {
	if a < b {
		return a
	}
	return b
}

func MaxI64(a, b int64) (c int64) {
	if a > b {
		return a
	}
	return b
}
func MinU(a, b uint) (c uint) {
	if a < b {
		return a
	}
	return b
}

func MaxU(a, b uint) (c uint) {
	if a > b {
		return a
	}
	return b
}
func MinU8(a, b uint8) (c uint8) {
	if a < b {
		return a
	}
	return b
}

func MaxU8(a, b uint8) (c uint8) {
	if a > b {
		return a
	}
	return b
}
func MinU16(a, b uint16) (c uint16) {
	if a < b {
		return a
	}
	return b
}

func MaxU16(a, b uint16) (c uint16) {
	if a > b {
		return a
	}
	return b
}
func MinU32(a, b uint32) (c uint32) {
	if a < b {
		return a
	}
	return b
}

func MaxU32(a, b uint32) (c uint32) {
	if a > b {
		return a
	}
	return b
}
func MinU64(a, b uint64) (c uint64) {
	if a < b {
		return a
	}
	return b
}

func MaxU64(a, b uint64) (c uint64) {
	if a > b {
		return a
	}
	return b
}
func MinF32(a, b float32) (c float32) {
	if a < b {
		return a
	}
	return b
}

func MaxF32(a, b float32) (c float32) {
	if a > b {
		return a
	}
	return b
}
func MinF64(a, b float64) (c float64) {
	if a < b {
		return a
	}
	return b
}

func MaxF64(a, b float64) (c float64) {
	if a > b {
		return a
	}
	return b
}
func MinStr(a, b string) (c string) {
	if a < b {
		return a
	}
	return b
}

func MaxStr(a, b string) (c string) {
	if a > b {
		return a
	}
	return b
}
func MinIterSVI(a int, b []int, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSI(a []int, b int, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterI(a, b []int, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVI(a int, b []int, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSI(a []int, b int, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterI(a, b []int, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVI8(a int8, b []int8, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSI8(a []int8, b int8, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterI8(a, b []int8, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVI8(a int8, b []int8, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSI8(a []int8, b int8, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterI8(a, b []int8, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVI16(a int16, b []int16, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSI16(a []int16, b int16, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterI16(a, b []int16, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVI16(a int16, b []int16, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSI16(a []int16, b int16, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterI16(a, b []int16, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVI32(a int32, b []int32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSI32(a []int32, b int32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterI32(a, b []int32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVI32(a int32, b []int32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSI32(a []int32, b int32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterI32(a, b []int32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVI64(a int64, b []int64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSI64(a []int64, b int64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterI64(a, b []int64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVI64(a int64, b []int64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSI64(a []int64, b int64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterI64(a, b []int64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVU(a uint, b []uint, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSU(a []uint, b uint, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterU(a, b []uint, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVU(a uint, b []uint, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSU(a []uint, b uint, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterU(a, b []uint, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVU8(a uint8, b []uint8, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSU8(a []uint8, b uint8, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterU8(a, b []uint8, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVU8(a uint8, b []uint8, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSU8(a []uint8, b uint8, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterU8(a, b []uint8, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVU16(a uint16, b []uint16, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSU16(a []uint16, b uint16, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterU16(a, b []uint16, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVU16(a uint16, b []uint16, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSU16(a []uint16, b uint16, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterU16(a, b []uint16, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVU32(a uint32, b []uint32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSU32(a []uint32, b uint32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterU32(a, b []uint32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVU32(a uint32, b []uint32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSU32(a []uint32, b uint32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterU32(a, b []uint32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVU64(a uint64, b []uint64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSU64(a []uint64, b uint64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterU64(a, b []uint64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVU64(a uint64, b []uint64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSU64(a []uint64, b uint64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterU64(a, b []uint64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVF32(a float32, b []float32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSF32(a []float32, b float32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterF32(a, b []float32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVF32(a float32, b []float32, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSF32(a []float32, b float32, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterF32(a, b []float32, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVF64(a float64, b []float64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSF64(a []float64, b float64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterF64(a, b []float64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVF64(a float64, b []float64, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSF64(a []float64, b float64, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterF64(a, b []float64, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MinIterSVStr(a string, b []string, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a < b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MinIterVSStr(a []string, b string, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b < a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMinIterStr(a, b []string, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] < a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}

func MaxIterSVStr(a string, b []string, bit Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if a > b[i] {
				b[i] = a
			}
		}
	}
	return
}

func MaxIterVSStr(a []string, b string, ait Iterator) (err error) {
	var i int
	var validi bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			if b > a[i] {
				a[i] = b
			}
		}
	}
	return
}

func VecMaxIterStr(a, b []string, ait, bit Iterator) (err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			if b[j] > a[i] {
				a[i] = b[j]
			}
		}
	}
	return
}
