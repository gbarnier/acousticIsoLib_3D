// Sources setup for Nonlinear modeling
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::setSources_3D(std::shared_ptr<deviceGpu_3D> sources){
	_sources = sources;
	_nSourcesReg = _sources->getNDeviceReg();
	_sourcesPositionReg = _sources->getRegPosUnique();
}

// Sources setup for Born and Tomo
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::setSources_3D(std::shared_ptr<deviceGpu_3D> sourcesDevices, std::shared_ptr<double2DReg> sourcesSignals){

	// Set source devices
	_sources = sourcesDevices;
	_nSourcesReg = _sources->getNDeviceReg();
	_sourcesPositionReg = _sources->getRegPosUnique();

	// Set source signals
	_sourcesSignals = sourcesSignals->clone(); // Source signal read from the input file (raw)
	_sourcesSignalsRegDts = std::make_shared<double2DReg>(_fdParam_3D->_nts, _nSourcesReg); // Source signal interpolated to the regular grid
	_sourcesSignalsRegDtsDt2 = std::make_shared<double2DReg>(_fdParam_3D->_nts, _nSourcesReg); // Source signal with second-order time derivative
	_sourcesSignalsRegDtwDt2 = std::make_shared<double2DReg>(_fdParam_3D->_ntw, _nSourcesReg); // Source signal with second-order time derivative on fine time-sampling grid
	_sourcesSignalsRegDtw = std::make_shared<double2DReg>(_fdParam_3D->_ntw, _nSourcesReg); // Source signal on fine time-sampling grid

	// Interpolate spatially to regular grid
	_sources->adjoint(false, _sourcesSignalsRegDts, _sourcesSignals); // Interpolate sources signals to regular grid

	// Apply second time derivative to sources signals
	_secTimeDer->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2);

	// Scale seismic source
	scaleSeismicSource_3D(_sources, _sourcesSignalsRegDtsDt2, _fdParam_3D); // Scale sources signals by dtw^2 * vel^2
	scaleSeismicSource_3D(_sources, _sourcesSignalsRegDts, _fdParam_3D); // Scale sources signals by dtw^2 * vel^2

	// Interpolate to fine time-sampling
	_timeInterp_3D->forward(false, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2); // Interpolate sources signals to fine time-sampling
	_timeInterp_3D->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtw); // Interpolate sources signals to fine time-sampling

}

// Receivers setup for Nonlinear modeling
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::setReceivers_3D(std::shared_ptr<deviceGpu_3D> receivers){
	_receivers = receivers;
	_nReceiversReg = _receivers->getNDeviceReg();
	_receiversPositionReg = _receivers->getRegPosUnique();
}

// Receivers setup for nonlinear modeling
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::setAcquisition_3D(std::shared_ptr<deviceGpu_3D> sources, std::shared_ptr<deviceGpu_3D> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources_3D(sources);
	setReceivers_3D(receivers);
	this->setDomainRange(model, data);
	assert(checkParfileConsistency_3D(model, data));
}

// Set acquisiton for Born and Tomo
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::setAcquisition_3D(std::shared_ptr<deviceGpu_3D> sources, std::shared_ptr<double2DReg> sourcesSignals, std::shared_ptr<deviceGpu_3D> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources_3D(sources, sourcesSignals);
	setReceivers_3D(receivers);
	this->setDomainRange(model, data);
	assert(checkParfileConsistency_3D(model, data));
}

// Scale seismic source
template <class V1, class V2>
void seismicOperator_3D <V1, V2>::scaleSeismicSource_3D(const std::shared_ptr<deviceGpu_3D> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParam_3D> parObj) const{

	std::shared_ptr<double2D> sig = signal->_mat;
	double *v = _fdParam_3D->_vel->getVals();
	long long *pos = seismicSource->getRegPosUnique();

	#pragma omp parallel for
	for (int iGridPoint = 0; iGridPoint < seismicSource->getNDeviceReg(); iGridPoint++){
		double scale = _fdParam_3D->_dtw * _fdParam_3D->_dtw * v[pos[iGridPoint]]*v[pos[iGridPoint]];
		for (int it = 0; it < signal->getHyper()->getAxis(1).n; it++){
			(*sig)[iGridPoint][it] = (*sig)[iGridPoint][it] * scale;
		}
	}
}
