/** 
*  @file   larpandoracontent/LArWorkshop/ClassificationAlgorithm.cc 
* 
*  @brief  Implementation of the Classification algorithm class. 
* 
*  $Log: $ */
#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/MyArea/ClassificationAlgorithm.h"
#include "PandoraMonitoringApi.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
//#include <image_ops.h>
//#include "/tensorflow/core/lib/core/status.h"
#include "tensorflow/c/c_api.h"

using namespace pandora;
namespace lar_content
{
StatusCode ClassificationAlgorithm::Run()
{
	std::cout << "Point0" <<std::endl;
	const CaloHitList *pCaloHitListU(nullptr);
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));
	std::cout << "Point0.1" <<std::endl;
	const CaloHitList *pCaloHitListV(nullptr);
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameV, pCaloHitListV));
	std::cout << "Point0.2" <<std::endl;
	const CaloHitList *pCaloHitListW(nullptr);
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameW, pCaloHitListW));
	std::cout << "Point0.3" << pCaloHitListU->size() << pCaloHitListV->size() << pCaloHitListW->size() <<std::endl;
	CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
	CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
	CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());
	std::cout << "Point1" <<std::endl;
	std::ostringstream tempStr;
	(void) HitsToStringStream(caloHitVectorU, tempStr);
	(void) HitsToStringStream(caloHitVectorV, tempStr);
	(void) HitsToStringStream(caloHitVectorW, tempStr);

	const std::string pathToHelper = "~/Pandora/PandoraPFA/LArContent-v03_15_04/larpandoracontent/MyArea/"; // Test setup to call a python helper function with the CNN
	std::string data = tempStr.str();
	data = data.substr(0, data.size()-1); // Removes last comma 																// TO DO: Import tensorflow model directly into C++ ... 
	std::cout << "Calling CNN Python Helper" <<std::endl;
	const std::string command = "python3 "+pathToHelper+"CNNHelper.py '"+ data + "'"; 											// ... and rasterize data here
	system(command.c_str());

	return STATUS_CODE_SUCCESS;
}

StatusCode ClassificationAlgorithm::HitsToStringStream(const CaloHitVector caloHitVector, std::ostringstream &tempStr)
{
	for (const CaloHit *const pCaloHit : caloHitVector)
	{
		tempStr << pCaloHit->GetPositionVector().GetX() << "," << pCaloHit->GetPositionVector().GetZ() << ",";
	}
	for(int i=0; i<2*(500-caloHitVector.size()); i++) // Pads with zeros
		tempStr << "0.0,";
	return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------
StatusCode ClassificationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
// Read settings from xml file here
//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameU", m_caloHitListNameU));
PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameV", m_caloHitListNameV));
PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameW", m_caloHitListNameW));

return STATUS_CODE_SUCCESS;
}
} // namespace lar_content
