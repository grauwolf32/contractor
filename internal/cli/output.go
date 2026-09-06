package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"text/tabwriter"
)

type OutputMode string

const (
	OutputTable OutputMode = "table"
	OutputJSON  OutputMode = "json"
	OutputName  OutputMode = "name"
)

type Printer struct {
	mode   OutputMode
	output io.Writer
}

func NewPrinter(mode string, output io.Writer) (*Printer, error) {
	parsed := OutputMode(mode)
	if parsed != OutputTable && parsed != OutputJSON && parsed != OutputName {
		return nil, errors.New("output must be table, json, or name")
	}
	return &Printer{mode: parsed, output: output}, nil
}

func (p *Printer) Mode() OutputMode { return p.mode }

func (p *Printer) JSON(value any) error {
	encoder := json.NewEncoder(p.output)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func (p *Printer) Names(values ...string) error {
	for _, value := range values {
		if _, err := fmt.Fprintln(p.output, value); err != nil {
			return err
		}
	}
	return nil
}

func (p *Printer) Table(headers []string, rows [][]string) error {
	writer := tabwriter.NewWriter(p.output, 0, 4, 2, ' ', 0)
	for index, header := range headers {
		if index != 0 {
			_, _ = io.WriteString(writer, "\t")
		}
		_, _ = io.WriteString(writer, header)
	}
	_, _ = io.WriteString(writer, "\n")
	for _, row := range rows {
		for index, column := range row {
			if index != 0 {
				_, _ = io.WriteString(writer, "\t")
			}
			_, _ = io.WriteString(writer, column)
		}
		_, _ = io.WriteString(writer, "\n")
	}
	return writer.Flush()
}

func (p *Printer) Object(value any, names ...string) error {
	if p.mode == OutputName {
		return p.Names(names...)
	}
	return p.JSON(value)
}
