package performance

import "time"

type GPUAggregate struct {
	Freshness Freshness         `json:"freshness"`
	Devices   []GPUDeviceGauges `json:"devices"`
}

type GPUDeviceGauges struct {
	ID                 string        `json:"id"`
	Name               string        `json:"name"`
	UtilizationPercent *GaugeSummary `json:"utilizationPercent,omitempty"`
	MemoryUsedBytes    *GaugeSummary `json:"memoryUsedBytes,omitempty"`
	MemoryTotalBytes   *GaugeSummary `json:"memoryTotalBytes,omitempty"`
	TemperatureCelsius *GaugeSummary `json:"temperatureCelsius,omitempty"`
	PowerWatts         *GaugeSummary `json:"powerWatts,omitempty"`
	PowerLimitWatts    *GaugeSummary `json:"powerLimitWatts,omitempty"`
}

func gpuGauges(d *GPUDeviceGauges) []**GaugeSummary {
	return []**GaugeSummary{&d.UtilizationPercent, &d.MemoryUsedBytes, &d.MemoryTotalBytes, &d.TemperatureCelsius, &d.PowerWatts, &d.PowerLimitWatts}
}

func gpuAggregateDevice(target *GPUAggregate, id, name string) *GPUDeviceGauges {
	for i := range target.Devices {
		if target.Devices[i].ID == id {
			return &target.Devices[i]
		}
	}
	if len(target.Devices) == MaxGPUDevices {
		reason := RecordLimit
		target.Freshness.Status, target.Freshness.Reason = Partial, &reason
		return nil
	}
	target.Devices = append(target.Devices, GPUDeviceGauges{ID: id, Name: name})
	return &target.Devices[len(target.Devices)-1]
}

func accumulateGPU(target **GPUAggregate, source *GPU) {
	if *target == nil {
		*target = &GPUAggregate{Devices: []GPUDeviceGauges{}}
	}
	g := *target
	g.Freshness = source.Freshness
	if source.Freshness.ObservedAt == nil {
		return
	}
	for _, source := range source.Devices {
		d := gpuAggregateDevice(g, source.ID, source.Name)
		if d == nil {
			continue
		}
		for _, pair := range []struct {
			target **GaugeSummary
			value  *float64
		}{
			{&d.UtilizationPercent, source.UtilizationPercent}, {&d.TemperatureCelsius, source.TemperatureCelsius},
			{&d.PowerWatts, source.PowerWatts}, {&d.PowerLimitWatts, source.PowerLimitWatts},
		} {
			if pair.value != nil {
				addGauge(pair.target, *pair.value, *g.Freshness.ObservedAt)
			}
		}
		for _, pair := range []struct {
			target **GaugeSummary
			value  *uint64
		}{
			{&d.MemoryUsedBytes, source.MemoryUsedBytes}, {&d.MemoryTotalBytes, source.MemoryTotalBytes},
		} {
			if pair.value != nil {
				addGauge(pair.target, float64(*pair.value), *g.Freshness.ObservedAt)
			}
		}
	}
}

func mergeGPU(target **GPUAggregate, source *GPUAggregate) {
	if source == nil {
		return
	}
	if *target == nil {
		*target = &GPUAggregate{Devices: []GPUDeviceGauges{}}
	}
	g := *target
	g.Freshness = source.Freshness
	for _, source := range source.Devices {
		d := gpuAggregateDevice(g, source.ID, source.Name)
		if d == nil {
			continue
		}
		for i, gauge := range gpuGauges(&source) {
			mergeGauge(gpuGauges(d)[i], *gauge)
		}
	}
}

func (g GPUAggregate) Validate(until time.Time) error {
	if g.Freshness.Validate() != nil || g.Freshness.IntervalSeconds != 15 || g.Freshness.LastAttemptAt.After(until) || g.Devices == nil || len(g.Devices) > MaxGPUDevices {
		return errInvalidRecord
	}
	seen := map[string]bool{}
	for _, d := range g.Devices {
		if !validGPUIdentity(d.ID, d.Name) || seen[d.ID] {
			return errInvalidRecord
		}
		seen[d.ID] = true
		for i, field := range gpuGauges(&d) {
			v := *field
			if v == nil {
				continue
			}
			if v.Samples == 0 || v.ObservedAt.IsZero() || v.ObservedAt.After(until) || v.Min > v.Last || v.Last > v.Max {
				return errInvalidRecord
			}
			if i == 0 && v.Max > 100 {
				return errInvalidRecord
			}
			if (i == 1 || i == 2) && v.Max > 1<<53-1 {
				return errInvalidRecord
			}
		}
	}
	return nil
}
