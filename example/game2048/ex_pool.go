package main

import (
	"bytes"
	"math/rand"
	"strconv"
)

type ExPool struct {
	len int64

	data []Ex
}

func NewExPool(maxLength int64) *ExPool {
	p := new(ExPool)
	p.len = maxLength
	p.data = make([]Ex, 0, maxLength)
	return p
}

func (p *ExPool) Len() int {
	return len(p.data)
}

func (p *ExPool) Push(ex Ex) {
	if len(p.data) == cap(p.data) {
		p.data = deleteSlice(p.data, 0)
	}

	p.data = append(p.data, ex)
}

func (p *ExPool) Sample(num int) []Ex {
	r := make([]Ex, 0, num)
	for {
		if len(r) == num {
			break
		}
		idx := rand.Intn(len(p.data))
		r = append(r, p.data[idx])
	}

	return r
}

func (p *ExPool) Optimal(state []uint) Ex {
	var max int
	ex := Ex{State: state}

	for i, item := range p.data {
		if item.StateKey() == ex.StateKey() && item.Reward > p.data[max].Reward {
			max = i
		}
	}

	return p.data[max]
}

type Ex struct {
	State     []uint
	Action    uint8
	Reward    float64
	NextState []uint
	End       bool
}

func (e *Ex) StateKey() string {
	var buf bytes.Buffer

	for i, item := range e.State {
		if i != 0 {
			buf.WriteString(",")
		}
		buf.WriteString(strconv.Itoa(int(item)))
	}

	return buf.String()
}

func deleteSlice(exs []Ex, idx int) []Ex {
	j := 0
	for i, v := range exs {
		if i != idx {
			exs[j] = v
			j++
		}
	}

	return exs[:j]
}
