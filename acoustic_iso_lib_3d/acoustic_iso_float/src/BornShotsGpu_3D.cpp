#include <vector>
#include <omp.h>
#include "BornShotsGpu_3D.h"
#include "BornGpu_3D.h"
#include <time.h>

BornShotsGpu_3D::BornShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector){

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
		allocatePinnedBornGpu_3D(_par->getInt("nz"), _par->getInt("nx"), _par->getInt("ny"), _par->getInt("nts"), _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory" << std::endl;
}

// Constructor for Fwime
BornShotsGpu_3D::BornShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::shared_ptr<tomoExtShotsGpu_3D> tomoExtGpuObj){

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
	std::cout << "Allocating source wavefields on pinned memory for Fwime Born" << std::endl;
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Allocating wavefield # " << iGpu << std::endl;
		setPinnedBornGpuFwime_3D(tomoExtGpuObj->getPinWavefieldVec()[iGpu], _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory for Fwime Born" << std::endl;
}

BornShotsGpu_3D::BornShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinsu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu){

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

	// _srcWavefieldVector = srcWavefieldVector;
	// Allocate wavefields on pinned memory
	std::cout << "Allocating source wavefields on pinned memory" << std::endl;
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Allocating wavefield # " << iGpu << std::endl;
		allocatePinnedBornGpu_3D(zAxisWavefield.n, xAxisWavefield.n, yAxisWavefield.n, timeAxis.n, _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory" << std::endl;

}

// Constructor for Ginsu + Fwime
BornShotsGpu_3D::BornShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinsu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu, std::shared_ptr<tomoExtShotsGpu_3D> tomoExtGpuObj){

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
	// axis zAxisWavefield = axis(_vel->getHyper()->getAxis(1).n, 1.0, 1.0);
	// axis xAxisWavefield = axis(nxMaxGinsu, 1.0, 1.0);
	// axis yAxisWavefield = axis(nyMaxGinsu, 1.0, 1.0);
	// axis timeAxis = _sourcesSignals->getHyper()->getAxis(1);

	_velHyperVectorGinsu = velHyperVectorGinsu;
	_xPadMinusVectorGinsu = xPadMinusVectorGinsu;
	_xPadPlusVectorGinsu = xPadPlusVectorGinsu;
	_ixVectorGinsu = ixVectorGinsu;
	_iyVectorGinsu = iyVectorGinsu;

	// Allocate wavefields on pinned memory
	std::cout << "Allocating source wavefields on pinned memory" << std::endl;
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Allocating wavefield # " << iGpu << std::endl;
		setPinnedBornGpuFwime_3D(tomoExtGpuObj->getPinWavefieldVec()[iGpu], _gpuList.size(), iGpu, _gpuList[iGpu], _iGpuAlloc);
	}
	std::cout << "Done allocating source wavefields on pinned memory" << std::endl;
}

// Gpu list
void BornShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR [BornShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl; throw std::runtime_error("");}

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
			std::cout << "**** ERROR [BornShotsGpu_3D]: Please make sure there are no duplicates in the list ****" << std::endl; throw std::runtime_error("");
		}
	}

	// Check that the user does not ask for more GPUs than shots to be modeled
	if (_nGpu > _nShot){std::cout << "**** ERROR [BornShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; throw std::runtime_error("");}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

// Gpu list
void BornShotsGpu_3D::deallocatePinnedBornGpu_3D(){

	// Deallocate pinned memory
	for (int iGpu=0; iGpu<_gpuList.size(); iGpu++){
		std::cout << "Deallocating wavefield # " << iGpu << std::endl;
		deallocatePinnedBornShotsGpu_3D(iGpu, _gpuList[iGpu]);
	}
}

// Forward
void BornShotsGpu_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		// std::cout << "Constant receiver geometry over shots" << std::endl;
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<float2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornGpu_3D>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu_3D> BornGpuObject(new BornGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornGpuObject->getFdParam_3D()->getInfo_3D();
		}

		if (_ginsu == 0){
			// Allocate memory on device
			allocateBornShotsGpu_3D(BornObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);
		}

		// Create data slice for this GPU number
		std::shared_ptr<SEP::float2DReg> dataSlice(new SEP::float2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Born forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::float2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// Temporary arrays/hypercube for the Ginsu
		std::shared_ptr<SEP::float3DReg> modelTemp;

		// If no ginsu is used, use the full model
		if (_ginsu == 0){
			modelTemp = model;

		// If ginsu is used, copy the part of the model into modelTemp
		} else {

			// Allocate and set Ginsu
			BornObjectVector[iGpu]->setBornGinsuGpu_3D(_velHyperVectorGinsu[iShot], (*_xPadMinusVectorGinsu->_mat)[iShot], (*_xPadPlusVectorGinsu->_mat)[iShot], _ixVectorGinsu[iShot], _iyVectorGinsu[iShot], iGpu, iGpuId);
			// Allocate Ginsu model
			modelTemp = std::make_shared<SEP::float3DReg>(_velHyperVectorGinsu[iShot]);

			// Temporary variables (for clarity purpose)
			int fat = BornObjectVector[iGpu]->getFdParam_3D()->_fat;
			int nzGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nzGinsu;
			int nxGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nxGinsu;
			int nyGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nyGinsu;
			int izGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_izGinsu;
			int ixGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_ixGinsu;
			int iyGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_iyGinsu;

			// Copy values into Ginsu model
			modelTemp->scale(0.0);
			#pragma omp parallel for collapse(3)
			for (int iy = fat; iy < nyGinsu-fat; iy++){
				for (int ix = fat; ix < nxGinsu-fat; ix++){
					for (int iz = 	fat; iz < nzGinsu-fat; iz++){
						(*modelTemp->_mat)[iy][ix][iz] = (*model->_mat)[iy+iyGinsu][ix+ixGinsu][iz+izGinsu];
					}
				}
			}
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<float2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(float)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}

		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<float2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(float)*_sourcesSignals->getHyper()->getAxis(1).n);
			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);

		// Apply forward Born
		BornObjectVector[iGpu]->forward(false, modelTemp, dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}
		if (_ginsu == 1){
			// Deallocate memory on GPU
			deallocateBornShotsGpu_3D(iGpu, iGpuId);
		}
	}

	if (_ginsu == 0){
		// Deallocate memory on device
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateBornShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}
}

// Adjoint
void BornShotsGpu_3D::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignals->getHyper()->getAxis(2).n == 1) {
		constantSrcSignal = 1;
	} else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		constantRecGeom=1;
	} else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<float3DReg>> modelSliceVector;
	std::vector<std::shared_ptr<float2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornGpu_3D>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu_3D> BornGpuObject(new BornGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornGpuObject->getFdParam_3D()->getInfo_3D();
		}

		if (_ginsu == 0){
			// Allocate memory on device for that object
			allocateBornShotsGpu_3D(BornObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam_3D()->_reflectivityScale, iGpu, _gpuList[iGpu]);
		}

		// Model slice
		std::shared_ptr<SEP::float3DReg> modelSlice(new SEP::float3DReg(hyperModelSlice));
		modelSlice->scale(0.0); // Check that
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::float2DReg> dataSlice(new SEP::float2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born adjoint
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		// std::cout << "Adj1, iShot = " << iShot << std::endl;

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Case where the source signature is not constant over shots
		std::shared_ptr<SEP::float2DReg> sourcesSignalsTemp;
		axis dummyAxis(1);

		// std::cout << "Adj2, iShot = " << iShot << std::endl;
		// std::cout << "hyperDataSlice->getAxis(1).n = " << hyperDataSlice->getAxis(1).n << std::endl;
		// std::cout << "hyperDataSlice->getAxis(2).n = " << hyperDataSlice->getAxis(2).n << std::endl;

		// Copy data slice
		long long int data_idx = iShot*hyperDataSlice->getAxis(1).n;
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[data_idx*hyperDataSlice->getAxis(2).n]), sizeof(float)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);
		// std::cout << "Adj 2.5, iShot = " << iShot << std::endl;

		// Temporary arrays/hypercube for the Ginsu
		std::shared_ptr<SEP::float3DReg> modelTemp;

		// std::cout << "Adj 2.75, iShot = " << iShot << std::endl;

		// If no ginsu is used, use the full model
		if (_ginsu == 0){
			modelTemp = model;

		// std::cout << "Adj3, iShot = " << iShot << std::endl;

		// If ginsu is used, copy the part of the model into modelTemp
		} else {

			// Allocate and set Ginsu
			BornObjectVector[iGpu]->setBornGinsuGpu_3D(_velHyperVectorGinsu[iShot], (*_xPadMinusVectorGinsu->_mat)[iShot], (*_xPadPlusVectorGinsu->_mat)[iShot], _ixVectorGinsu[iShot], _iyVectorGinsu[iShot], iGpu, iGpuId);

			// Allocate Ginsu model
			modelTemp = std::make_shared<SEP::float3DReg>(_velHyperVectorGinsu[iShot]);

			// Set model temp to zero
			modelTemp->scale(0.0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			// std::cout << "Adj4, iShot = " << iShot << std::endl;
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<float2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(float)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[0], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			// std::cout << "Adj5, iShot = " << iShot << std::endl;
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _sourcesSignals, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {

			// Create a 2D-temporary array where you store the wavelet for this shot
			sourcesSignalsTemp = std::make_shared<float2DReg>(_sourcesSignals->getHyper()->getAxis(1),dummyAxis);
    		memcpy(sourcesSignalsTemp->getVals(), &(_sourcesSignals->getVals()[iShot*_sourcesSignals->getHyper()->getAxis(1).n]), sizeof(float)*_sourcesSignals->getHyper()->getAxis(1).n);

			// Set the acquisition geometry (shot+receiver) for this specific shot
			BornObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], sourcesSignalsTemp, _receiversVector[iShot], modelTemp, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);

		// Launch modeling
		if (_ginsu == 0){
			// std::cout << "iShot = " << iShot << std::endl;
			// std::cout << "iGpu = " << iGpu << std::endl;
			// Launch adjoint
			BornObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);
			// std::cout << "Done iShot = " << iShot << std::endl;
		} else {
			// Launch adjoint
			BornObjectVector[iGpu]->adjoint(false, modelTemp, dataSliceVector[iGpu]);

			// Copy modeTemp into modelSliceVector[iGpu]
			// Temporary variables (for clarity purpose)
			int fat = BornObjectVector[iGpu]->getFdParam_3D()->_fat;
			int nzGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nzGinsu;
			int nxGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nxGinsu;
			int nyGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_nyGinsu;
			int izGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_izGinsu;
			int ixGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_ixGinsu;
			int iyGinsu = BornObjectVector[iGpu]->getFdParam_3D()->_iyGinsu;

			// Copy values into Ginsu model
			#pragma omp parallel for collapse(3)
			for (int iy = fat; iy < nyGinsu-fat; iy++){
				for (int ix = fat; ix < nxGinsu-fat; ix++){
					for (int iz = 	fat; iz < nzGinsu-fat; iz++){
						(*modelSliceVector[iGpu]->_mat)[iy+iyGinsu][ix+ixGinsu][iz+izGinsu] += (*modelTemp->_mat)[iy][ix][iz];
					}
				}
			}
			// Deallocate memory on Gpu
			deallocateBornShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}

	// std::cout << "Done adjoint, stacking all shots" << std::endl;
	// Stack models computed by each GPU
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
		// std::cout << "Stacking shots from gpu#" << iGpu << std::endl;
        #pragma omp parallel for
        for (int iy=0; iy<model->getHyper()->getAxis(3).n; iy++){
            for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
                for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
                    (*model->_mat)[iy][ix][iz] += (*modelSliceVector[iGpu]->_mat)[iy][ix][iz];
                }
            }
        }
    }

	// std::cout << "Done adjoint" << std::endl;
	// Deallocate memory on device
	if (_ginsu == 0){
		// std::cout << "Done adjoint + deallocation" << std::endl;
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateBornShotsGpu_3D(iGpu, _gpuList[iGpu]);
		}
	}
}
