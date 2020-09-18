#include <vector>
#include <omp.h>
#include "BornExtShotsGpu_3D.h"
#include "BornExtGpu_3D.h"

BornExtShotsGpu_3D::BornExtShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector){

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList_3D();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	if (not getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo)){
		throw std::runtime_error("Error in getGpuInfo_3D");
   	}
	_ginsu = par->getInt("ginsu");
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignals = sourcesSignals;

	// Allocate wavefields on pinned memory
	std::cout << "Allocating source wavefields on pinned memory" << std::endl;
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Allocating wavefield # " << iGpu << std::endl;
		allocatePinnedBornExtGpu_3D(_par->getInt("nz"), _par->getInt("nx"), _par->getInt("ny"), _par->getInt("nts"), _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory" << std::endl;
}

BornExtShotsGpu_3D::BornExtShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinsu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu){

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList_3D();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	if ( not getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo) ){
		throw std::runtime_error("Error in getGpuInfo");
   	}
	_ginsu = par->getInt("ginsu");
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignals = sourcesSignals;

	// Compute the dimension of the largest model we will have to allocate among all the different shots
	axis zAxisWavefield = axis(_vel->getHyper()->getAxis(1).n, 1.0, 1.0);
	axis xAxisWavefield = axis(nxMaxGinsu, 1.0, 1.0);
	axis yAxisWavefield = axis(nyMaxGinsu, 1.0, 1.0);
	axis timeAxis = _sourcesSignals->getHyper()->getAxis(1);

	_velHyperVectorGinsu = velHyperVectorGinsu;
	_xPadMinusVectorGinsu = xPadMinusVectorGinsu;
	_xPadPlusVectorGinsu = xPadPlusVectorGinsu;
	_ixVectorGinsu = ixVectorGinsu;
	_iyVectorGinsu = iyVectorGinsu;

	// Allocate wavefields on pinned memory
	std::cout << "Allocating source wavefields on pinned memory" << std::endl;
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Allocating wavefield # " << iGpu << std::endl;
		allocatePinnedBornExtGpu_3D(zAxisWavefield.n, xAxisWavefield.n, yAxisWavefield.n, timeAxis.n, _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory" << std::endl;

}

void BornExtShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR [BornExtShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl; throw std::runtime_error("");}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR [BornExtShotsGpu_3D]: Please make sure there are no duplicates in the list ****" << std::endl; throw std::runtime_error("");
		}
	}

	// Check that the user does not ask for more GPUs than shots to be modeled
	if (_nGpu > _nShot){std::cout << "**** ERROR [BornExtShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; throw std::runtime_error("");}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

// Gpu list
void BornExtShotsGpu_3D::deallocatePinnedBornExtGpu_3D(){

	// Deallocate pinned memory
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Deallocating wavefield # " << iGpu << std::endl;
		deallocatePinnedBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
	}
}

void BornExtShotsGpu_3D::forward(const bool add, const std::shared_ptr<double5DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {
			// std::cout << "Constant source signal over shots" << std::endl;
			constantSrcSignal = 1; }
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		// std::cout << "Constant receiver geometry over shots" << std::endl;
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornExtGpu_3D>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornExtGpu_3D> BornExtGpuObject(new BornExtGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam_3D()->getInfo_3D();
		}

		if (_ginsu == 0){
			// Allocate memory on device
			allocateBornExtShotsGpu_3D(BornExtObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);
		}

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::double2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// Temporary arrays/hypercube for the Ginsu
		std::shared_ptr<SEP::double5DReg> modelTemp;

		// If no ginsu is used, use the full model
		if (_ginsu == 0){
			modelTemp = model;

		// If ginsu is used, copy the part of the model into modelTemp
		} else {

			// Allocate and set Ginsu
			BornExtObjectVector[iGpu]->setBornExtGinsuGpu_3D(_velHyperVectorGinsu[iShot], (*_xPadMinusVectorGinsu->_mat)[iShot], (*_xPadPlusVectorGinsu->_mat)[iShot], _ixVectorGinsu[iShot], _iyVectorGinsu[iShot], iGpu, iGpuId);
			// std::cout << "_velHyperVectorGinsu[iShot] n1 = " << _velHyperVectorGinsu[iShot]->getAxis(1).n << std::endl;
			// std::cout << "_velHyperVectorGinsu[iShot] n2 = " << _velHyperVectorGinsu[iShot]->getAxis(2).n << std::endl;
			// std::cout << "_velHyperVectorGinsu[iShot] n3 = " << _velHyperVectorGinsu[iShot]->getAxis(3).n << std::endl;
			// std::cout << "_velHyperVectorGinsu[iShot] n4 = " << _velHyperVectorGinsu[iShot]->getAxis(4).n << std::endl;
			// std::cout << "_velHyperVectorGinsu[iShot] n5 = " << _velHyperVectorGinsu[iShot]->getAxis(5).n << std::endl;
			// Allocate Ginsu model

			// Temporary variables (for clarity purpose)
			int fat = BornExtObjectVector[iGpu]->getFdParam_3D()->_fat;
			int nzGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nzGinsu;
			int nxGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nxGinsu;
			int nyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nyGinsu;
			int izGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_izGinsu;
			int ixGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_ixGinsu;
			int iyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_iyGinsu;
			int nExt1 = BornExtObjectVector[iGpu]->getFdParam_3D()->_nExt1;
			int nExt2 = BornExtObjectVector[iGpu]->getFdParam_3D()->_nExt2;
			axis zAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_zAxisGinsu;
			axis xAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_xAxisGinsu;
			axis yAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_yAxisGinsu;
			axis extAxis1Ginsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_extAxis1;
			axis extAxis2Ginsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_extAxis2;

			modelTemp = std::make_shared<SEP::double5DReg>(zAxisGinsu, xAxisGinsu, yAxisGinsu, extAxis1Ginsu, extAxis2Ginsu);

			// Copy values into Ginsu model
			modelTemp->scale(0.0);
			#pragma omp parallel for collapse(5)
			for (int iExt2 = 0; iExt2 < nExt2; iExt2++){
				for (int iExt1 = 0; iExt1 < nExt1; iExt1++){
					for (int iy = fat; iy < nyGinsu-fat; iy++){
						for (int ix = fat; ix < nxGinsu-fat; ix++){
							for (int iz = 	fat; iz < nzGinsu-fat; iz++){
								(*modelTemp->_mat)[iExt2][iExt1][iy][ix][iz] = (*model->_mat)[iExt2][iExt1][iy+iyGinsu][ix+ixGinsu][iz+izGinsu];
							}
						}
					}
				}
			}
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);
			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);
		// std::cout << "Born modelTemp min before = " << modelTemp->min() << std::endl;
		// std::cout << "Born modelTemp max before = " << modelTemp->max() << std::endl;
		BornExtObjectVector[iGpu]->forward(false, modelTemp, dataSliceVector[iGpu]);
		// std::cout << "Born dataSliceVector[iGpu] min after = " << dataSliceVector[iGpu]->min() << std::endl;
		// std::cout << "Born dataSliceVector[iGpu] max after = " << dataSliceVector[iGpu]->max() << std::endl;

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

		if (_ginsu == 1){
			// Deallocate memory on GPU
			deallocateBornExtShotsGpu_3D(iGpu, iGpuId);
		}
	}

	if (_ginsu == 0){
		// Deallocate memory on device
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}

	// std::cout << "Born extended forward, min data = " << data->min() << std::endl;
	// std::cout << "Born extended forward, max data = " << data->max() << std::endl;

}

void BornExtShotsGpu_3D::adjoint(const bool add, std::shared_ptr<double5DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {
		std::cout << "Constant source signal over shots" << std::endl;
		constantSrcSignal = 1;}
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3), model->getHyper()->getAxis(4), model->getHyper()->getAxis(5)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double5DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornExtGpu_3D>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornExtGpu_3D> BornExtGpuObject(new BornExtGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam_3D()->getInfo_3D();
		}

		if (_ginsu == 0){
			// Allocate memory on device for that object
			allocateBornExtShotsGpu_3D(BornExtObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);
		}

		// Model slice
		std::shared_ptr<SEP::double5DReg> modelSlice(new SEP::double5DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born extended adjoint
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::double2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Temporary arrays/hypercube for the Ginsu
		std::shared_ptr<SEP::double5DReg> modelTemp;
		std::shared_ptr<SEP::hypercube> hyperTemp;

		// If no ginsu is used, use the full model
		if (_ginsu == 0){
			modelTemp = model;

		// If ginsu is used, copy the part of the model into modelTemp
		} else {

			// Allocate and set Ginsu
			BornExtObjectVector[iGpu]->setBornExtGinsuGpu_3D(_velHyperVectorGinsu[iShot], (*_xPadMinusVectorGinsu->_mat)[iShot], (*_xPadPlusVectorGinsu->_mat)[iShot], _ixVectorGinsu[iShot], _iyVectorGinsu[iShot], iGpu, iGpuId);

			// Temporary variables (for clarity purpose)
			int fat = BornExtObjectVector[iGpu]->getFdParam_3D()->_fat;
			int nzGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nzGinsu;
			int nxGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nxGinsu;
			int nyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nyGinsu;
			int izGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_izGinsu;
			int ixGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_ixGinsu;
			int iyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_iyGinsu;
			int nExt1 = BornExtObjectVector[iGpu]->getFdParam_3D()->_nExt1;
			int nExt2 = BornExtObjectVector[iGpu]->getFdParam_3D()->_nExt2;
			axis zAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_zAxisGinsu;
			axis xAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_xAxisGinsu;
			axis yAxisGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_yAxisGinsu;
			axis extAxis1Ginsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_extAxis1;
			axis extAxis2Ginsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_extAxis2;

			modelTemp = std::make_shared<SEP::double5DReg>(zAxisGinsu, xAxisGinsu, yAxisGinsu, extAxis1Ginsu, extAxis2Ginsu);

			// Copy values into Ginsu model
			modelTemp->scale(0.0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<double2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(double)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornExtObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);

		// Launch modeling
		if (_ginsu==0){
			// Launch adjoint
			// std::cout << "Shots ext adj" << std::endl;
			// std::cout << "dataSliceVector[iGpu] max = " << dataSliceVector[iGpu]->max() << std::endl;
			// std::cout << "dataSliceVector[iGpu] min = " << dataSliceVector[iGpu]->min() << std::endl;
			BornExtObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);
			// std::cout << "modelSliceVector[iGpu] max = " << modelSliceVector[iGpu]->max() << std::endl;
			// std::cout << "modelSliceVector[iGpu] min = " << modelSliceVector[iGpu]->min() << std::endl;
		} else {
			// Launch adjoint
			BornExtObjectVector[iGpu]->adjoint(false, modelTemp, dataSliceVector[iGpu]);

			// Copy modeTemp into modelSliceVector[iGpu]
			// Temporary variables (for clarity purpose)
			int fat = BornExtObjectVector[iGpu]->getFdParam_3D()->_fat;
			int nzGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nzGinsu;
			int nxGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nxGinsu;
			int nyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_nyGinsu;
			int izGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_izGinsu;
			int ixGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_ixGinsu;
			int iyGinsu = BornExtObjectVector[iGpu]->getFdParam_3D()->_iyGinsu;

			// Copy values into Ginsu model
			#pragma omp parallel for collapse(5)
			for (int iExt2=0; iExt2<model->getHyper()->getAxis(5).n; iExt2++){
				for (int iExt1=0; iExt1<model->getHyper()->getAxis(4).n; iExt1++){
					for (int iy = fat; iy < nyGinsu-fat; iy++){
						for (int ix = fat; ix < nxGinsu-fat; ix++){
							for (int iz = 	fat; iz < nzGinsu-fat; iz++){
								(*modelSliceVector[iGpu]->_mat)[iExt2][iExt1][iy+iyGinsu][ix+ixGinsu][iz+izGinsu] += (*modelTemp->_mat)[iExt2][iExt1][iy][ix][iz];
							}
						}
					}
				}
			}
			// Deallocate memory on Gpu
			deallocateBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}

	// Stack models computed by each GPU
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
        #pragma omp parallel for collapse(5)
		for (int iExt2=0; iExt2<model->getHyper()->getAxis(5).n; iExt2++){
			for (int iExt1=0; iExt1<model->getHyper()->getAxis(4).n; iExt1++){
        		for (int iy=0; iy<model->getHyper()->getAxis(3).n; iy++){
            		for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
                		for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
                    		(*model->_mat)[iExt2][iExt1][iy][ix][iz] += (*modelSliceVector[iGpu]->_mat)[iExt2][iExt1][iy][ix][iz];
                		}
            		}
        		}
			}
		}
	}

	// Deallocate memory on device
	if (_ginsu == 0){
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateBornExtShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}
	//
	// std::cout << "Born extended adjoint, min model = " << model->min() << std::endl;
	// std::cout << "Born extended adjoint, max model = " << model->max() << std::endl;
}
