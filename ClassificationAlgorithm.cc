/** 
*  @file   larpandoracontent/LArWorkshop/ClassificationAlgorithm.cc
* 
*  @brief  Implementation of the TrainingExport algorithm class. 
* 
*  $Log: $ */
#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/MyArea/ClassificationAlgorithm.h"
#include "PandoraMonitoringApi.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArInteractionTypeHelper.h"
#include "larpandoracontent/LArHelpers/LArMonitoringHelper.h"

#include <fstream>
#include <stdlib.h>


using namespace pandora;
namespace lar_content
{

StatusCode ClassificationAlgorithm::Run()
{
	const CaloHitList *pCaloHitListU(nullptr);
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));

	CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
	std::sort(caloHitVectorU.begin(), caloHitVectorU.end(), LArClusterHelper::SortHitsByPosition);
	std::stringstream tempStr;
	for (const CaloHit *const pCaloHit : caloHitVectorU)
	{
		tempStr << pCaloHit->GetPositionVector().GetX() << "," 
				<< pCaloHit->GetPositionVector().GetZ() << ",";
	}
	std:string data = (tempStr.str()).pop_back();
	system("python3 CNNHelper.py '"+data+"'"); // Calls the python program that does all the data handling and prints out the prediction to the terminal
												// This is not an ideal implementation but implementing the data to image conversion in C++ would require more work  
	
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

return EventValidationAlgorithm::ReadSettings(xmlHandle);
}
} // namespace lar_content
